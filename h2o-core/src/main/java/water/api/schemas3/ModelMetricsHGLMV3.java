package water.api.schemas3;

import hex.ModelMetricsHGLM;
import water.api.API;

public class ModelMetricsHGLMV3<I extends ModelMetricsHGLM, S extends ModelMetricsHGLMV3<I, S>>
        extends ModelMetricsBaseV3<I, S> {
  @API(help="standard error of fixed predictors/effects", direction=API.Direction.OUTPUT)
  public double[] seFe;       // standard error of fixed predictors/effects

  @API(help="standard error of random effects", direction=API.Direction.OUTPUT)  
  public double[] seRe;       // standard error of random effects

  @API(help="dispersion parameter of the mean model (residual variance for LMM)", direction=API.Direction.OUTPUT)
  public double varFix;       // dispersion parameter of the mean model (residual variance for LMM)

  @API(help="dispersion parameter of the random effects (variance of random effects for GLMM", direction=API.Direction.OUTPUT)
  public double[] varRanef;     // dispersion parameter of the random effects (variance of random effects for GLMM)

  @API(help="fixed coefficient)", direction=API.Direction.OUTPUT)
  public double fixef;       // dispersion parameter of the mean model (residual variance for LMM)

  @API(help="random coefficients", direction=API.Direction.OUTPUT)
  public double[] ranef;     // dispersion parameter of the random effects (variance of random effects for GLMM)
 
  @API(help="true if model has converged", direction=API.Direction.OUTPUT)
  public boolean converge;    // true if model has converged

  @API(help="number of random columns", direction=API.Direction.OUTPUT)
  public int nRandC;       // deviance degrees of freedom for mean part of the model

  @API(help="deviance degrees of freedom for mean part of the model", direction=API.Direction.OUTPUT)
  public double dfReFe;       // deviance degrees of freedom for mean part of the model

  @API(help="estimates, standard errors of the linear predictor in the dispersion model", direction=API.Direction.OUTPUT)  
  public double[] summVC1;    // estimates, standard errors of the linear predictor in the dispersion model

  @API(help="estimates, standard errors of the linear predictor for dispersion parameter of random effects", direction=API.Direction.OUTPUT)
  public double[][] summVC2;// estimates, standard errors of the linear predictor for dispersion parameter of random effects

  @API(help="log h-likelihood", direction=API.Direction.OUTPUT)
  public double hlik;         // log h-likelihood

  @API(help="adjusted profile log-likelihood profiled over random effects", direction=API.Direction.OUTPUT)  
  public double pvh;          // adjusted profile log-likelihood profiled over random effects

  @API(help="adjusted profile log-likelihood profiled over fixed and random effects", direction=API.Direction.OUTPUT)
  public double pbvh;         // adjusted profile log-likelihood profiled over fixed and random effects

  @API(help="conditional AIC", direction=API.Direction.OUTPUT)
  public double cAIC;         // conditional AIC
  
  @API(help="index of the most influential observation", direction=API.Direction.OUTPUT)
  public long  bad;           // index of the most influential observation
  
  @API(help="sum(etai-eta0)^2 where etai is current eta and eta0 is the previous one", direction=API.Direction.OUTPUT)
  public double sumEtaDiffSquare;  // sum(etai-eta0)^2
  
  @API(help="sum(etai-eta0)^2/sum(etai)^2 ", direction=API.Direction.OUTPUT)
  public double convergence;       // sum(etai-eta0)^2/sum(etai)^2
  
  @Override
  public S fillFromImpl(ModelMetricsHGLM modelMetrics) {
    super.fillFromImpl(modelMetrics);
    hlik = modelMetrics._hlik;
    pvh = modelMetrics._pvh;
    pbvh = modelMetrics._pbvh;
    cAIC = modelMetrics._cAIC;
    bad = modelMetrics._bad;
    sumEtaDiffSquare=modelMetrics._sumEtaDiffSquare;
    convergence = modelMetrics._convergence;
    nRandC = modelMetrics._nRandC;
    varFix = modelMetrics._varFix;
    varRanef = modelMetrics._varRanef;
    converge = modelMetrics._converge;
    dfReFe = modelMetrics._dfReFe;    
    seFe = new double[modelMetrics._seFe.length];
    System.arraycopy(modelMetrics._seFe, 0, seFe, 0, seFe.length);
    seRe = new double[modelMetrics._seRe.length];
    System.arraycopy(modelMetrics._seRe, 0, seRe, 0, seRe.length);
    varRanef = new double[modelMetrics._varRanef.length];
    System.arraycopy(modelMetrics._varRanef, 0, varRanef, 0, varRanef.length);
    summVC1 = new double[modelMetrics._summVC1.length];
    System.arraycopy(modelMetrics._summVC1, 0, summVC1, 0, summVC1.length);
    summVC2 = new double[nRandC][];
    for (int index=0; index < nRandC; index++) {
      int l = modelMetrics._summVC2[index].length;
      summVC2[index] = new double[l];
      System.arraycopy(modelMetrics._summVC2[index], 0, summVC2[index], 0, l);
    }
    return (S) this;
  }
}
