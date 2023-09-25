
import sys,array,math,os
import pickle
import numpy as np
import tensorflow as tf
import narf
import narf.fitutils
import narf.tfutils


def get_scaled_eigenvectors(hess, num_null = 2):
    e,v = np.linalg.eigh(hess)

    # remove the null eigenvectors
    e = e[None, num_null:]
    v = v[:, num_null:]

    # scale the eigenvectors
    vscaled = v/np.sqrt(e)

    return vscaled

def get_invcdf(input_file):

    ut_axis = 1
    qt_axis = 0

    handle = open(input_file, 'rb')
    res = pickle.load(handle)
    quant_cdfvals = res['quant_cdfvals_tf']
    knots_qt = res['knots_qt']
    order = res['order']
    extrpl = res['extrpl']
    parms = tf.constant(res['x'], tf.float64)
    vscaled = tf.constant(get_scaled_eigenvectors(res['hess'], num_null=0 if res['withConstraint'] else res['order']), tf.float64)
    ut_low = tf.cast(res['recoilMin'], tf.float64)
    ut_high = tf.cast(res['recoilMax'], tf.float64)
    def func_invcdf(pt, quant):
        pts = tf.reshape(pt, (1,1))
        quant_outs = tf.reshape(quant, (1,1))

        parms_2d = tf.reshape(parms, (-1, order))
        parms_2d = tf.transpose(parms_2d)
        qparms = narf.fitutils.cubic_spline_interpolate(knots_qt, parms_2d, pts, axis=qt_axis, extrpl=extrpl)
        quants = narf.fitutils.qparms_to_quantiles(qparms, x_low=ut_low, x_high=ut_high, axis=ut_axis)
        cdfinvvals = narf.fitutils.pchip_interpolate(quant_cdfvals, quants, quant_outs, axis=ut_axis)
        return cdfinvvals

    return parms, vscaled, func_invcdf, ut_low, ut_high

def get_cdf(input_file):

    ut_axis = 1
    qt_axis = 0

    handle = open(input_file, 'rb')
    res = pickle.load(handle)
    quant_cdfvals = res['quant_cdfvals_tf']
    knots_qt = res['knots_qt']
    order = res['order']
    extrpl = res['extrpl']
    parms = tf.constant(res['x'], tf.float64)
    vscaled = tf.constant(get_scaled_eigenvectors(res['hess'], num_null=0 if res['withConstraint'] else res['order']), tf.float64)
    ut_low = tf.cast(res['recoilMin'], tf.float64)
    ut_high = tf.cast(res['recoilMax'], tf.float64)
    def func_cdf(pt, ut):
        pts = tf.reshape(pt, (1,1))
        uts = tf.reshape(ut, (1,1))

        parms_2d = tf.reshape(parms, (-1, order))
        parms_2d = tf.transpose(parms_2d)
        qparms = narf.fitutils.cubic_spline_interpolate(knots_qt, parms_2d, pts, axis=qt_axis, extrpl=extrpl)
        quants = narf.fitutils.qparms_to_quantiles(qparms, x_low=ut_low, x_high=ut_high, axis=ut_axis)
        cdfvals = narf.fitutils.pchip_interpolate(quants, quant_cdfvals, uts, axis=ut_axis)
        return cdfvals

    return parms, vscaled, func_cdf, ut_low, ut_high


def exportModel(target_para, target_perp, source_para, source_perp, out):

    parms_target_para, vscaled_target_para, func_cdfinv_target_para, ut_low_target_para, ut_high_target_para = get_invcdf(target_para)
    parms_target_perp, vscaled_target_perp, func_cdfinv_target_perp, ut_low_target_perp, ut_high_target_perp = get_invcdf(target_perp)

    parms_source_para, vscaled_source_para, func_cdf_source_para, ut_low_source_para, ut_high_source_para = get_cdf(source_para)
    parms_source_perp, vscaled_source_perp, func_cdf_source_perp, ut_low_source_perp, ut_high_source_perp = get_cdf(source_perp)

    nparms_para = parms_target_para.shape[0] + parms_source_para.shape[0]
    nparms_perp = parms_target_perp.shape[0] + parms_source_perp.shape[0]

    def func_cdfinv_pdf_target_para(pt, quant):
        with tf.GradientTape() as t:
            t.watch(quant)
            cdfinv = func_cdfinv_target_para(pt, quant)
        pdfreciprocal = t.gradient(cdfinv, quant)
        pdf = 1./pdfreciprocal
        return cdfinv, pdf

    def func_cdfinv_pdf_target_perp(pt, quant):
        with tf.GradientTape() as t:
            t.watch(quant)
            cdfinv = func_cdfinv_target_perp(pt, quant)
        pdfreciprocal = t.gradient(cdfinv, quant)
        pdf = 1./pdfreciprocal
        return cdfinv, pdf

    def transform_para_no_unc(pt, ut_para):
        cdf_mc = func_cdf_source_para(pt, ut_para)
        ut_transformed, pdf = func_cdfinv_pdf_target_para(pt, cdf_mc)
        ut_transformed = tf.reshape(ut_transformed, [])
        return ut_transformed
        
    def transform_perp_no_unc(pt, ut_perp):
        cdf_mc = func_cdf_source_perp(pt, ut_perp)
        ut_transformed, pdf = func_cdfinv_pdf_target_perp(pt, cdf_mc)
        ut_transformed = tf.reshape(ut_transformed, [])
        return ut_transformed

    def transform_para(pt, ut_para):

        with tf.GradientTape(persistent=True) as t:
            t.watch(parms_source_para)
            t.watch(parms_target_para)

            cdf_source_para = func_cdf_source_para(pt, ut_para)
            ut_transformed_para, pdf_para = func_cdfinv_pdf_target_para(pt, cdf_source_para)

            ut_transformed_para = tf.reshape(ut_transformed_para, [])
            pdf_para = tf.reshape(pdf_para, [])

        pdf_grad_source_para = t.gradient(pdf_para, parms_source_para)
        pdf_grad_target_para = t.gradient(pdf_para, parms_target_para)

        del t

        weight_grad_source_para = pdf_grad_source_para/pdf_para
        weight_grad_target_para = pdf_grad_target_para/pdf_para

        weight_grad_source_para = weight_grad_source_para[None, :]
        weight_grad_target_para = weight_grad_target_para[None, :]

        weight_grad_mc_eig_para = weight_grad_source_para @ vscaled_source_para
        weight_grad_data_eig_para = weight_grad_target_para @ vscaled_target_para

        weight_grad_mc_eig_para = tf.reshape(weight_grad_mc_eig_para, [-1])
        weight_grad_data_eig_para = tf.reshape(weight_grad_data_eig_para, [-1])

        weight_grad_eig_para = tf.concat([weight_grad_mc_eig_para, weight_grad_data_eig_para], axis=0) + 1.0
        
        return ut_transformed_para, weight_grad_eig_para
        
    def transform_perp(pt, ut_perp):

        with tf.GradientTape(persistent=True) as t:
            t.watch(parms_source_perp)
            t.watch(parms_target_perp)

            cdf_source_perp = func_cdf_source_perp(pt, ut_perp)
            ut_transformed_perp, pdf_perp = func_cdfinv_pdf_target_perp(pt, cdf_source_perp)

            ut_transformed_perp = tf.reshape(ut_transformed_perp, [])
            pdf_perp = tf.reshape(pdf_perp, [])

        pdf_grad_source_perp = t.gradient(pdf_perp, parms_source_perp)
        pdf_grad_target_perp = t.gradient(pdf_perp, parms_target_perp)

        del t

        weight_grad_source_perp = pdf_grad_source_perp/pdf_perp
        weight_grad_target_perp = pdf_grad_target_perp/pdf_perp

        weight_grad_source_perp = weight_grad_source_perp[None, :]
        weight_grad_target_perp = weight_grad_target_perp[None, :]

        weight_grad_mc_eig_perp = weight_grad_source_perp @ vscaled_source_perp
        weight_grad_data_eig_perp = weight_grad_target_perp @ vscaled_target_perp

        weight_grad_mc_eig_perp = tf.reshape(weight_grad_mc_eig_perp, [-1])
        weight_grad_data_eig_perp = tf.reshape(weight_grad_data_eig_perp, [-1])

        weight_grad_eig_perp = tf.concat([weight_grad_mc_eig_perp, weight_grad_data_eig_perp], axis=0) + 1.0

        return ut_transformed_perp, weight_grad_eig_perp

    def recoil_meta():
        return tf.constant(nparms_para, tf.int32), tf.constant(nparms_perp, tf.int32), ut_low_target_para, ut_high_target_para



    scalar_spec = tf.TensorSpec([], tf.float64)
    input_signature = [scalar_spec, scalar_spec]
    model = narf.tfutils.function_to_tflite([transform_para, transform_perp, transform_para_no_unc, transform_perp_no_unc, recoil_meta], [input_signature, input_signature, input_signature, input_signature, None])

    with open(out, 'wb') as f:
        f.write(model)


    test_interp = tf.lite.Interpreter(model_content=model)
    print(test_interp.get_input_details())
    print(test_interp.get_output_details())
    print(test_interp.get_signature_list())

    meta = test_interp.get_signature_runner('recoil_meta')()
    signatures_out = test_interp.get_signature_list()['recoil_meta']['outputs']
    print()
    print()

    #print(outputs)
    nparms_para = meta[signatures_out[0]]
    nparms_perp = meta[signatures_out[1]]
    ut_low = meta[signatures_out[2]]
    ut_high = meta[signatures_out[3]]
    print(nparms_para)
    print(nparms_perp)
    print(ut_low)
    print(ut_high)


if __name__ == "__main__":

    tag = "highPU_DeepMETReso"

    target_para = f"data/{tag}/singlemuon_para_postfit.pkl"
    target_perp = f"data/{tag}/singlemuon_perp_postfit.pkl"
    source_para = f"data/{tag}/zmumu_para_postfit.pkl"
    source_perp = f"data/{tag}/zmumu_perp_postfit.pkl"
    out = f"data/{tag}/model_data_mc.tflite"
    exportModel(target_para, target_perp, source_para, source_perp, out)
    quit()

    target_para = f"data/{tag}/zmumu_para_postfit.pkl"
    target_perp = f"data/{tag}/zmumu_perp_postfit.pkl"
    source_para = f"data/{tag}/zmumu_gen_para_postfit.pkl"
    source_perp = f"data/{tag}/zmumu_gen_perp_postfit.pkl"
    out = f"data/{tag}/model_mc_gen.tflite"
    exportModel(target_para, target_perp, source_para, source_perp, out)
