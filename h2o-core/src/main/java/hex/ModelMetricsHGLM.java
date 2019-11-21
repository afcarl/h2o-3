package hex;

import water.exceptions.H2OIllegalArgumentException;
import water.fvec.Frame;

import java.util.Arrays;

public class ModelMetricsHGLM extends ModelMetricsSupervised {
  public final double[] _seFe;       // standard error of fixed predictors/effects
  public final double[] _seRe;       // standard error of random effects
  public final double[] _fixedf;     // fixed coefficients
  public final double[] _ranef;       // random coefficients
  public final int _nRandC;           // number of random columns in Z
  public final double _varFix;       // dispersion parameter of the mean model (residual variance for LMM)
  public final double[] _varRanef;     // dispersion parameter of the random effects (variance of random effects for GLMM)
  public final boolean _converge;    // true if model has converged
  public final double _dfReFe;       // deviance degrees of freedom for mean part of the model
  public final double[] _summVC1;    // estimates, standard errors of the linear predictor in the dispersion model
  public final double[][] _summVC2;// estimates, standard errors of the linear predictor for dispersion parameter of random effects
  public final double _hlik;         // log h-likelihood
  public final double _pvh;          // adjusted profile log-likelihood profiled over random effects
  public final double _pbvh;         // adjusted profile log-likelihood profiled over fixed and random effects
  public final double _cAIC;         // conditional AIC
  public final long  _bad;           // index of the most influential observation
  public final double _sumEtaDiffSquare;  // sum(etai-eta0)^2
  public final double _convergence;       // sum(etai-eta0)^2/sum(etai)^2
  public final int _iterations;
  
  public ModelMetricsHGLM(Model model, Frame frame, long nobs, double mse, String[] domain, double sigma, 
                          CustomMetric customMetric, double[] sefe, double[] sere, double varfix, double[] varranef,
                          boolean converge, double dfrefe, double[] summvc1, double[][] summvc2, double hlik, 
                          double pvh, double pbvh, double cAIC, long bad, double sumEtaDiffSq, double convergence, 
                          int randC, double[] fixef, double[] ranef, int iter) {
    super(model, frame, nobs, mse, domain, sigma, customMetric);
    _seFe = sefe;
    _seRe = sere;
    _varFix = varfix;
    _varRanef = varranef;
    _converge = converge;
    _dfReFe = dfrefe;
    _summVC1 = summvc1;
    _summVC2 = summvc2;
    _hlik = hlik;
    _pvh = pvh;
    _pbvh = pbvh;
    _cAIC = cAIC;
    _bad = bad;
    _sumEtaDiffSquare = sumEtaDiffSq;
    _convergence = convergence;
    _nRandC = randC;
    _fixedf = fixef;
    _ranef = ranef;
    _iterations = iter;
  }

  public static ModelMetricsHGLM getFromDKV(Model model, Frame frame) {
    ModelMetrics mm = ModelMetrics.getFromDKV(model, frame);
    if( !(mm instanceof ModelMetricsHGLM) )
      throw new H2OIllegalArgumentException("Expected to find a HGLM ModelMetrics for model: " + model._key.toString()
              + " and frame: " + frame._key.toString(), "Expected to find a ModelMetricsHGLM for model: " + 
              model._key.toString() + " and frame: " + frame._key.toString() + " but found a: " + (mm == null ? null : mm.getClass()));
    return (ModelMetricsHGLM) mm;
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append(super.toString());
    sb.append(" standard error of fixed predictors effects: "+Arrays.toString(_seFe));
    sb.append(" standard error of random effects: "+Arrays.toString(_seRe));
    sb.append(" dispersion parameter of the mean model (residual variance for LMM): "+_varFix);
    sb.append(" dispersion parameter of the random effects (variance of random effects for GLMM): "+_varRanef);
    if (_converge)
      sb.append(" HGLM has converged.");
    else
      sb.append(" HGLM has failed to converge.");
    sb.append(" deviance degrees of freedom for mean part of the model: "+_dfReFe);
    sb.append(" estimates, standard errors of the linear predictor in the dispersion model: "+Arrays.toString(_summVC1));
    sb.append(" estimates, standard errors of the linear predictor for dispersion parameter of random effects: "+
            Arrays.toString(_summVC2));
    sb.append(" log h-likelihood: "+_hlik);
    sb.append(" adjusted profile log-likelihood profiled over random effects: "+_pvh);
    sb.append(" adjusted profile log-likelihood profiled over fixed and random effects: "+_pbvh);
    sb.append(" conditional AIC: "+_cAIC);
    sb.append(" index of the most influential observation: "+_bad);
    sb.append(" sum(etai-eta0)^2: "+_sumEtaDiffSquare);
    sb.append("convergence (sum(etai-eta0)^2/sum(etai)^2): "+_convergence);
    return sb.toString();
  }
  
  public static class MetricBuilderHGLM<T extends MetricBuilderHGLM<T>> extends MetricBuilderSupervised<T> {
    public double[] _seFe;       // standard error of fixed predictors/effects
    public double[] _seRe;       // standard error of random effects
    public double _varFix;       // dispersion parameter of the mean model (residual variance for LMM)
    public double[] _varRanef;     // dispersion parameter of the random effects (variance of random effects for GLMM)
    public boolean _converge;    // true if model has converged
    public double _dfReFe;       // deviance degrees of freedom for mean part of the model
    public double[] _summVC1;    // estimates, standard errors of the linear predictor in the dispersion model
    public double[][] _summVC2;// estimates, standard errors of the linear predictor for dispersion parameter of random effects
    public double _hlik;         // log h-likelihood
    public double _pvh;          // adjusted profile log-likelihood profiled over random effects
    public double _pbvh;         // adjusted profile log-likelihood profiled over fixed and random effects
    public double _cAIC;         // conditional AIC
    public long  _bad;           // index of the most influential observation
    public double _sumEtaDiffSquare;  // sum(etai-eta0)^2
    public double _convergence;       // sum(etai-eta0)^2/sum(etai)^2
    public double[] _fixf;
    public double[] _ranef;
    public int _nRandC;
    public int _iterations;    // number of iterations
    public long _nobs;

    public MetricBuilderHGLM(String[] domain) {
      super(0,domain);
    }
    
    public void updateCoeffs(double[] fixedCoeffs, double[] randCoeffs) {
      int fixfLen = fixedCoeffs.length;
      if (_fixf==null) 
        _fixf = new double[fixfLen];
      System.arraycopy(fixedCoeffs, 0, _fixf, 0, fixfLen);
      
      int randLen = randCoeffs.length;
      if (_ranef == null)
        _ranef = new double[randLen];
      System.arraycopy(_ranef, 0, randCoeffs, 0, randLen);
      
    }

    public void updateSummVC(double[] VC1, double[][] VC2) {
      if (_summVC1==null)
        _summVC1 = new double[2];
      System.arraycopy(VC1, 0, _summVC1, 0, 2);
      
      if (_summVC2 == null) {
        _nRandC = VC2.length;
        _summVC2 = new double[_nRandC][2];
      }
      
      for (int index=0; index < _nRandC; index++)
        System.arraycopy(VC2[index], 0, _summVC2[index], 0, 2);
    }
    
    @Override
    public double[] perRow(double[] ds, float[] yact, Model m) {
      return new double[0];
    }

    @Override
    public ModelMetrics makeModelMetrics(Model m, Frame f, Frame adaptedFrame, Frame preds) {
      ModelMetricsHGLM mm = new ModelMetricsHGLM(m, f, _nobs, 0, _domain, 0, _customMetric, _seFe, _seRe, 
              _varFix, _varRanef, _converge, _dfReFe, _summVC1, _summVC2, _hlik, _pvh, _pbvh, _cAIC, _bad, 
              _sumEtaDiffSquare, _convergence, _nRandC, _fixf, _ranef, _iterations);
      if (m!=null) m.addModelMetrics(mm);
      return mm;
    }
  }
}
