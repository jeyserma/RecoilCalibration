
import sys,array,math,os
import pickle
import numpy as np
import tensorflow as tf
import narf
import narf.fitutils



def get_scaled_eigenvectors(hess, num_null = 2):
    e,v = np.linalg.eigh(hess)

    # remove the null eigenvectors
    e = e[None, num_null:]
    v = v[:, num_null:]

    # scale the eigenvectors
    vscaled = v/np.sqrt(e)

    return vscaled

def exportModel(mc_in, data_in, out):

    ut_axis = 1
    qt_axis = 0

    # inv cdf data
    handle_data = open(data_in, 'rb')
    res_data = pickle.load(handle_data)
    quant_cdfvals_data = res_data['quant_cdfvals_tf']
    knots_qt_data = res_data['knots_qT']
    order_data = res_data['order']
    extrpl_data = res_data['extrpl']
    parms_data = tf.constant(res_data['x'], tf.float64)
    vscaled_data = tf.constant(get_scaled_eigenvectors(res_data['hess'], num_null=0 if res_data['withConstraint'] else res_data['order']), tf.float64)
    ut_low_data = tf.cast(res_data['recoilMin'], tf.float64)
    ut_high_data = tf.cast(res_data['recoilMax'], tf.float64)
    def func_cdfinv_data(pt, quant):
        pts = tf.reshape(pt, (1,1))
        quant_outs = tf.reshape(quant, (1,1))

        parms_2d = tf.reshape(parms_data, (-1, order_data))
        parms_2d = tf.transpose(parms_2d)
        qparms = narf.fitutils.cubic_spline_interpolate(knots_qt_data, parms_2d, pts, axis=qt_axis, extrpl=extrpl_data)
        quants = narf.fitutils.qparms_to_quantiles(qparms, x_low=ut_low_data, x_high=ut_high_data, axis=ut_axis)
        cdfinvvals = narf.fitutils.pchip_interpolate(quant_cdfvals_data, quants, quant_outs, axis=ut_axis)
        return cdfinvvals


    # cdf mc
    handle_mc = open(mc_in, 'rb')
    res_mc = pickle.load(handle_mc)
    quant_cdfvals_mc = res_mc['quant_cdfvals_tf']
    knots_qt_mc = res_mc['knots_qT']
    order_mc = res_mc['order']
    extrpl_mc = res_mc['extrpl']
    parms_mc = tf.constant(res_mc['x'], tf.float64)
    vscaled_mc = tf.constant(get_scaled_eigenvectors(res_mc['hess'], num_null=0 if res_mc['withConstraint'] else res_mc['order']), tf.float64)
    ut_low_mc = tf.cast(res_mc['recoilMin'], tf.float64)
    ut_high_mc = tf.cast(res_mc['recoilMax'], tf.float64)
    def func_cdf_mc(pt, ut):
        if ut > ut_high_mc:
            ut = ut_high_mc
        if ut < ut_low_mc:
            ut = ut_low_mc

        pts = tf.reshape(pt, (1,1))
        uts = tf.reshape(ut, (1,1))

        parms_2d = tf.reshape(parms_mc, (-1, order_mc))
        parms_2d = tf.transpose(parms_2d)
        qparms = narf.fitutils.cubic_spline_interpolate(knots_qt_mc, parms_2d, pts, axis=qt_axis, extrpl=extrpl_mc)
        quants = narf.fitutils.qparms_to_quantiles(qparms, x_low=ut_low_mc, x_high=ut_high_mc, axis=ut_axis)
        cdfvals = narf.fitutils.pchip_interpolate(quants, quant_cdfvals_mc, uts, axis=ut_axis)
        return cdfvals


    def func_cdfinv_pdf_data(pt, quant):
        with tf.GradientTape() as t:
            t.watch(quant)
            cdfinv = func_cdfinv_data(pt, quant)
        pdfreciprocal = t.gradient(cdfinv, quant)
        pdf = 1./pdfreciprocal
        return cdfinv, pdf

    def transform_mc_simple(pt, ut):
        cdf_mc = func_cdf_mc(pt, ut)
        ut_transformed, pdf = func_cdfinv_pdf_data(pt, cdf_mc)
        ut_transformed = tf.reshape(ut_transformed, [])
        return ut_transformed

    def transform_mc(pt, ut):
        with tf.GradientTape(persistent=True) as t:
            t.watch(parms_mc)
            t.watch(parms_data)

            cdf_mc = func_cdf_mc(pt, ut)
            ut_transformed, pdf = func_cdfinv_pdf_data(pt, cdf_mc)

            ut_transformed = tf.reshape(ut_transformed, [])
            pdf = tf.reshape(pdf, [])

        pdf_grad_mc = t.gradient(pdf, parms_mc)
        pdf_grad_data = t.gradient(pdf, parms_data)

        del t

        weight_grad_mc = pdf_grad_mc/pdf
        weight_grad_data = pdf_grad_data/pdf

        weight_grad_mc = weight_grad_mc[None, :]
        weight_grad_data = weight_grad_data[None, :]

        weight_grad_mc_eig = weight_grad_mc @ vscaled_mc
        weight_grad_data_eig = weight_grad_data @ vscaled_data

        weight_grad_mc_eig = tf.reshape(weight_grad_mc_eig, [-1])
        weight_grad_data_eig = tf.reshape(weight_grad_data_eig, [-1])

        weight_grad_eig = tf.concat([weight_grad_mc_eig, weight_grad_data_eig], axis=0)

        return ut_transformed, weight_grad_eig

    pt = tf.constant(50, tf.float64)
    ut = tf.constant(10, tf.float64)
    print(func_cdf_mc(pt, ut))

    pt = tf.constant(50, tf.float64)
    quant = tf.constant(0, tf.float64)
    print(func_cdfinv_data(pt, quant))

    pt = tf.constant(5, tf.float64)
    ut = tf.constant(10, tf.float64)
    print(transform_mc_simple(pt, ut))

    print(transform_mc(pt, ut))

    #quit()

    scalar_spec = tf.TensorSpec([], tf.float64)
    class TestMod(tf.Module):
        @tf.function(input_signature = [scalar_spec, scalar_spec])
        def __call__(self, pt, ut):
            return transform_mc(pt, ut)

    module = TestMod()
    concrete_function = module.__call__.get_concrete_function()

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function], module)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]

    tflite_model = converter.convert()
    with open(out, 'wb') as f:
        f.write(tflite_model)

    print("Done")



if __name__ == "__main__":

    exportModel("data/zmumu_perp_postfit_DeepMETReso.pkl", "data/data_perp_postfit_DeepMETReso.pkl", "data/recoil_perp_DeepMETReso.tflite")
