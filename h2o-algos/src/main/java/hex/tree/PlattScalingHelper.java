package hex.tree;

import hex.Model;
import hex.ModelBuilder;
import hex.ModelCategory;
import hex.glm.GLM;
import hex.glm.GLMModel;
import water.*;
import water.fvec.Frame;
import water.fvec.Vec;

import static hex.ModelCategory.Binomial;

public class PlattScalingHelper {
    
    public interface ModelBuilderWithCalibration<M extends Model<M , P, O>, P extends Model.Parameters, O extends Model.Output> {
        void warn(String field, String msg);
        void error(String field, String msg);
        int nclasses();
        Frame init_adaptFrameToTrain(Frame f, String label, String param, boolean expensive);
        Frame getCalibrationFrame();
        void setCalibrationFrame(Frame f);
        void setCalibrationOutput(GLM calibrationOutput, M model);
    }

    public interface ParamsWithCalibration {
        boolean isCVModel();
        String weightsColumn();
        String responseColumn();
        
        Frame getCalibrationFrame();
        boolean calibrateModel();
    }

    public interface OutputWithCalibration {
        ModelCategory getModelCategory();
        GLMModel calibrationModel();
    }
    
    public static void initCalibration(ModelBuilderWithCalibration builder, ParamsWithCalibration parms, boolean expensive) {
        // Calibration
        Frame cf = parms.getCalibrationFrame();  // User-given calibration set
        if (cf != null) {
            if (! parms.calibrateModel())
                builder.warn("_calibration_frame", "Calibration frame was specified but calibration was not requested.");
            Frame adaptedCf = builder.init_adaptFrameToTrain(cf, "Calibration Frame", "_calibration_frame", expensive);
            builder.setCalibrationFrame(adaptedCf);
        }
        if (parms.calibrateModel()) {
            if (builder.nclasses() != 2)
                builder.error("_calibrate_model", "Model calibration is only currently supported for binomial models.");
            if (cf == null)
                builder.error("_calibrate_model", "Calibration frame was not specified.");
        }
    }
    
    public static <M extends Model<M , P, O>, P extends Model.Parameters, O extends Model.Output> void buildCalibrationModel(
        boolean finalScoring, ModelBuilderWithCalibration<M, P, O> builder, 
        ParamsWithCalibration parms, Job job, M model
    ) {
        // Model Calibration (only for the final model, not CV models)
        if (finalScoring && parms.calibrateModel() && (!parms.isCVModel())) {
            Key<Frame> calibInputKey = Key.make();
            try {
                Scope.enter();
                job.update(0, "Calibrating probabilities");
                Frame calib = builder.getCalibrationFrame();
                Vec calibWeights = parms.weightsColumn() != null ? calib.vec(parms.weightsColumn()) : null;
                Frame calibPredict = Scope.track(model.score(calib, null, job, false));
                Frame calibInput = new Frame(calibInputKey,
                    new String[]{"p", "response"}, new Vec[]{calibPredict.vec(1), calib.vec(parms.responseColumn())});
                if (calibWeights != null) {
                    calibInput.add("weights", calibWeights);
                }
                DKV.put(calibInput);

                Key<Model> calibModelKey = Key.make();
                Job calibJob = new Job<>(calibModelKey, ModelBuilder.javaName("glm"), "Platt Scaling (GLM)");
                GLM calibBuilder = ModelBuilder.make("GLM", calibJob, calibModelKey);
                calibBuilder._parms._intercept = true;
                calibBuilder._parms._response_column = "response";
                calibBuilder._parms._train = calibInput._key;
                calibBuilder._parms._family = GLMModel.GLMParameters.Family.binomial;
                calibBuilder._parms._lambda = new double[] {0.0};
                if (calibWeights != null) {
                    calibBuilder._parms._weights_column = "weights";
                }

                builder.setCalibrationOutput(calibBuilder, model);
            } finally {
                Scope.exit();
                DKV.remove(calibInputKey);
            }
        }
    }

    public static Frame postProcessPredictions(Frame predictFr, Job j, OutputWithCalibration output) {
        if (output.calibrationModel() == null) {
            return predictFr;
        } else if (output.getModelCategory() == Binomial) {
            Key<Job> jobKey = j != null ? j._key : null;
            Key<Frame> calibInputKey = Key.make();
            Frame calibOutput = null;
            try {
                Frame calibInput = new Frame(calibInputKey, new String[]{"p"}, new Vec[]{predictFr.vec(1)});
                calibOutput = output.calibrationModel().score(calibInput);
                assert calibOutput._names.length == 3;
                Vec[] calPredictions = calibOutput.remove(new int[]{1, 2});
                // append calibrated probabilities to the prediction frame
                predictFr.write_lock(jobKey);
                for (int i = 0; i < calPredictions.length; i++)
                    predictFr.add("cal_" + predictFr.name(1 + i), calPredictions[i]);
                return predictFr.update(jobKey);
            } finally {
                predictFr.unlock(jobKey);
                DKV.remove(calibInputKey);
                if (calibOutput != null)
                    calibOutput.remove();
            }
        } else {
            throw H2O.unimpl("Calibration is only supported for binomial models");
        }
    }
}
