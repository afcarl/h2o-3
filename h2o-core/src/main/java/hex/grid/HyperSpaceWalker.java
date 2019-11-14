package hex.grid;

import hex.Model;
import hex.ModelParametersBuilderFactory;
import hex.ScoreKeeper;
import hex.ScoringInfo;
import hex.grid.filter.PermutationFilterFunction;
import hex.grid.filter.KeepOnlyFirstMatchFilterFunction;
import water.exceptions.H2OIllegalArgumentException;
import water.util.PojoUtils;

import java.util.*;

import static java.lang.StrictMath.min;

public interface HyperSpaceWalker<MP extends Model.Parameters, C extends HyperSpaceSearchCriteria> {

  abstract class HyperSpaceIterator<MP extends Model.Parameters> {
    /**
     * Get next model parameters.
     *
     * <p>It should return model parameters for next point in hyper space.
     * Throws {@link java.util.NoSuchElementException} if there is no remaining point in space
     * to explore.</p>
     *
     * <p>The method can optimize based on previousModel, but should be
     * able to handle null-value.</p>
     *
     * @param previousModel  model generated for the previous point in hyper space, can be null.
     *
     * @return model parameters for next point in hyper space or null if there is no such point.
     *
     * @throws IllegalArgumentException  when model parameters cannot be constructed
     * @throws java.util.NoSuchElementException if the iteration has no more elements
     */
    abstract public MP nextModelParameters(Model previousModel);

    /**
     * Returns true if the iterator can continue.  Takes into account strategy-specific stopping criteria, if any.
     * @param previousModel  optional parameter which helps to determine next step, can be null
     * @return  true if the iterator can produce one more model parameters configuration.
     */
    abstract public boolean hasNext(Model previousModel);

    abstract public void reset();

    /**
     * @return the total time allowed for building this grid, in seconds.
     */
    abstract public double max_runtime_secs();

    /**
     * @return the total time allowed for building this grid, in seconds.
     */
    abstract public int max_models();

    /**
     * @return the time remaining for building this grid, in seconds.
     */
    abstract public double time_remaining_secs();

    /**
     * Inform the Iterator that a model build failed in case it needs to adjust its internal state.
     * @param failedModel
     */
    abstract public void modelFailed(Model failedModel);

    /**
     * Returns current "raw" state of iterator.
     *
     * The state is represented by a permutation of values of grid parameters.
     *
     * @return  array of "untyped" values representing configuration of grid parameters
     */
    abstract public Object[] getCurrentRawParameters();

    /**
     * 
     * @param permutation single permutation from the grid, represented as array of Objects and ordered based on {@code hyperParamNames} array
     * @param hyperParamNames array of hyper parameters names grid search is performed over
     * @param filterFunctions list of {@link PermutationFilterFunction} to check against. 
     *                        Permutation will be skipped if any of the functions consider it as to be skipped one.
     */
    boolean permutationIsSkipped(Object[] permutation, String[] hyperParamNames, ArrayList<PermutationFilterFunction> filterFunctions) {
      boolean skipPermutation = false;

      Map<String, Object> permutationAsMap = convertToMap(permutation, hyperParamNames);

      for (PermutationFilterFunction fun : filterFunctions) {
        if (!fun.apply(permutationAsMap)) {
          skipPermutation = true;
          break;
        }
      }
      // Decrement counters only when we actually decided not to skip given grid item (after taking into account all filter functions),
      // and only for KeepOnlyFirstMatchFilterFunction functions that already considered given permutation as matching its conditions
      if (!skipPermutation) {
        filterFunctions.stream()
                .filter(fun1 -> {
                  return fun1 instanceof KeepOnlyFirstMatchFilterFunction && ((KeepOnlyFirstMatchFilterFunction) fun1)._baseMatchFunction.apply(permutationAsMap);
                })
                .forEach(fun -> ((KeepOnlyFirstMatchFilterFunction) fun).decrementCounter());
      }
      return skipPermutation;
    }

    /**
     * Changes permutation's representation from Object[] to Map<String, Object> for an ability
     * to specify filter functions with hyper parameters names, i.e. to make it more user friendly
     */
    private Map<String, Object> convertToMap(Object[] permutation, String[] hyperParamNames) {
      Map<String, Object> permutationAsMap = new HashMap<>();
      int indexOfHP = 0;
      for(String hpName : hyperParamNames) {
        permutationAsMap.put(hpName, permutation[indexOfHP]);
        indexOfHP++;
      }
      return permutationAsMap;
    }
  } // interface HyperSpaceIterator

  /**
   * Search criteria for the hyperparameter search including directives for how to search and
   * when to stop the search.
   */
  C search_criteria();

  /** Based on the last model, the given array of ScoringInfo, and our stopping criteria should we stop early? */
  boolean stopEarly(Model model, ScoringInfo[] sk);

  /**
   * Returns an iterator to traverse this hyper-space.
   *
   * @return an iterator
   */
  HyperSpaceIterator<MP> iterator();

  /**
   * Returns hyper parameters names which are used for walking the hyper parameters space.
   *
   * The names have to match the names of attributes in model parameters MP.
   *
   * @return names of used hyper parameters
   */
  String[] getHyperParamNames();

  /**
   * Return estimated maximum size of hyperspace, not subject to any early stopping criteria.
   *
   * Can return -1 if estimate is not available.
   *
   * @return size of hyper space to explore
   */
  long getMaxHyperSpaceSize();

  /**
   * Return initial model parameters for search.
   * @return  return model parameters
   */
  MP getParams();

  ModelParametersBuilderFactory<MP> getParametersBuilderFactory();

  /**
   * Superclass for for all hyperparameter space walkers.
   * <p>
   * The external Grid / Hyperparameter search API uses a HashMap<String,Object> to describe a set of hyperparameter
   * values, where the String is a valid field name in the corresponding Model.Parameter, and the Object is
   * the field value (boxed as needed).
   */
  abstract class BaseWalker<MP extends Model.Parameters, C extends HyperSpaceSearchCriteria> implements HyperSpaceWalker<MP, C> {

    /**
     * @see #search_criteria()
     */
    final protected C _search_criteria;

    /**
     * Search criteria for the hyperparameter search including directives for how to search and
     * when to stop the search.
     */
    public C search_criteria() { return _search_criteria; }

    /** Based on the last model, the given array of ScoringInfo, and our stopping criteria should we stop early? */
    @Override
    public boolean stopEarly(Model model, ScoringInfo[] sk) {
      return false;
    }

    /**
     * Parameters builder factory to create new instance of parameters.
     */
    final transient ModelParametersBuilderFactory<MP> _paramsBuilderFactory;

    /**
     * Used "base" model parameters for this grid search.
     * The object is used as a prototype to create model parameters
     * for each point in hyper space.
     */
    final MP _params;

    /**
     * Hyper space description - in this case only dimension and possible values.
     */
    final protected Map<String, Object[]> _hyperParamsGrid;

    protected boolean _set_model_seed_from_search_seed = false;  // true if model parameter seed is set to default value and false otherwise
    long model_number = 0l;   // denote model number
    /**
     * Cached names of used hyper parameters.
     */
    final protected String[] _hyperParamNames;

    /**
     * Compute max size of hyper space to walk. May include duplicates if points in space are specified multiple
     * times.
     */
    final protected long _maxHyperSpaceSize;

    /**
     * List of filter functions having a match from any of which is enough to discard given permutation
     */
    ArrayList<PermutationFilterFunction> _filterFunctions;

    /**
     * Gives an ability for {@link HyperSpaceWalker} to reset its {@code _filterFunctions}'s states
     * as some of those functions might be stateful 
     */
    void resetFilterFunctions(ArrayList<PermutationFilterFunction> filterFunctions) {
      if(filterFunctions != null) filterFunctions.stream()
              .filter(fun -> fun instanceof KeepOnlyFirstMatchFilterFunction)
              .forEach(fun -> ((KeepOnlyFirstMatchFilterFunction)fun).reset());
    }

    /**
     * Java hackery so we can have a factory method on a class with type params.
     */
    public static class WalkerFactory<MP extends Model.Parameters, C extends HyperSpaceSearchCriteria> {
      /**
       * Factory method to create an instance based on the given HyperSpaceSearchCriteria instance.
       */
      public static <MP extends Model.Parameters, C extends HyperSpaceSearchCriteria>
        HyperSpaceWalker<MP, ? extends HyperSpaceSearchCriteria> create(MP params,
                                                                        Map<String, Object[]> hyperParams,
                                                                        ModelParametersBuilderFactory<MP> paramsBuilderFactory,
                                                                        C search_criteria) {
        HyperSpaceSearchCriteria.Strategy strategy = search_criteria.strategy();

        if (strategy == HyperSpaceSearchCriteria.Strategy.Cartesian) {
          return new HyperSpaceWalker.CartesianWalker<>(params, hyperParams, paramsBuilderFactory, (HyperSpaceSearchCriteria.CartesianSearchCriteria) search_criteria);
        } else if (strategy == HyperSpaceSearchCriteria.Strategy.RandomDiscrete ) {
          return new HyperSpaceWalker.RandomDiscreteValueWalker<>(params, hyperParams, paramsBuilderFactory, (HyperSpaceSearchCriteria.RandomDiscreteValueSearchCriteria) search_criteria);
        } else {
          throw new H2OIllegalArgumentException("strategy", "GridSearch", strategy);
        }
      }
    }

    /**
     *
     * @param paramsBuilderFactory
     * @param hyperParamsGrid
     */
    public BaseWalker(MP params,
                      Map<String, Object[]> hyperParamsGrid,
                      ModelParametersBuilderFactory<MP> paramsBuilderFactory,
                      C search_criteria) {
      _params = params;
      _hyperParamsGrid = hyperParamsGrid;
      _paramsBuilderFactory = paramsBuilderFactory;
      _hyperParamNames = hyperParamsGrid.keySet().toArray(new String[0]);
      _maxHyperSpaceSize = computeMaxSizeOfHyperSpace();
      _search_criteria = search_criteria;

      // Sanity check the hyperParams map, and check it against the params object
      MP defaults = null;
      try {
        defaults = (MP) params.getClass().newInstance();
      }
      catch (Exception e) {
        throw new H2OIllegalArgumentException("Failed to instantiate a new Model.Parameters object to get the default values.");
      }

      // if a parameter is specified in both model parameter and hyper-parameter, this is only allowed if the
      // parameter value is set to be default.  Otherwise, an exception will be thrown.
      for (String key : hyperParamsGrid.keySet()) {
        // Throw if the user passed an empty value list:
        Object[] values = hyperParamsGrid.get(key);
        if (0 == values.length)
          throw new H2OIllegalArgumentException("Grid search hyperparameter value list is empty for hyperparameter: " + key);

        if ("seed".equals(key) || "_seed".equals(key)) continue;  // initialized to the wall clock

        // Ugh.  Java callers, like the JUnits or Sparkling Water users, use a leading _.  REST users don't.
        String prefix = (key.startsWith("_") ? "" : "_");

        // Throw if params has a non-default value which is not in the hyperParams map
        Object defaultVal = PojoUtils.getFieldValue(defaults, prefix + key, PojoUtils.FieldNaming.CONSISTENT);
        Object actualVal = PojoUtils.getFieldValue(params, prefix + key, PojoUtils.FieldNaming.CONSISTENT);

        if (defaultVal != null && actualVal != null) {
          // both are not set to null
          if (defaultVal.getClass().isArray() &&
              // array
              !PojoUtils.arraysEquals(defaultVal, actualVal)) {
              throw new H2OIllegalArgumentException("Grid search model parameter '" + key + "' is set in both the model parameters and in the hyperparameters map.  This is ambiguous; set it in one place or the other, not both.");
          } // array
          if (!defaultVal.getClass().isArray() &&
              // ! array
              !defaultVal.equals(actualVal)) {
            throw new H2OIllegalArgumentException("Grid search model parameter '" + key + "' is set in both the model parameters and in the hyperparameters map.  This is ambiguous; set it in one place or the other, not both.");
          } // ! array
        } // both are set: defaultVal != null && actualVal != null

        // defaultVal is null but actualVal is not, raise exception
        if (defaultVal == null && !(actualVal == null)) {
          // only actual is set
            throw new H2OIllegalArgumentException("Grid search model parameter '" + key + "' is set in both the model parameters and in the hyperparameters map.  This is ambiguous; set it in one place or the other, not both.");
        }
      } // for all keys

      // check model parameter seed value and determine if it is set to default value for random gridsearch
      if ((search_criteria != null) &&
              (search_criteria.strategy() == HyperSpaceSearchCriteria.Strategy.RandomDiscrete)) {
        Object defaultSeedVal = PojoUtils.getFieldValue(defaults, "_seed", PojoUtils.FieldNaming.CONSISTENT);
        Object actualSeedVal = PojoUtils.getFieldValue(params, "_seed", PojoUtils.FieldNaming.CONSISTENT);
        long gridSeed = ((HyperSpaceSearchCriteria.RandomDiscreteValueSearchCriteria) search_criteria).seed();

        if ((defaultSeedVal != null) && (actualSeedVal != null)) {
          if (defaultSeedVal.equals(actualSeedVal) && !defaultSeedVal.equals(gridSeed)) { // param seed = default, gridSeed != default
            _set_model_seed_from_search_seed = true;
          }
        }
      }
    } // BaseWalker()

    @Override
    public String[] getHyperParamNames() {
      return _hyperParamNames;
    }

    @Override
    public long getMaxHyperSpaceSize() {
      return _maxHyperSpaceSize;
    }

    @Override
    public MP getParams() {
      return _params;
    }

    @Override
    public ModelParametersBuilderFactory<MP> getParametersBuilderFactory() {
      return _paramsBuilderFactory;
    }

    protected MP getModelParams(MP params, Object[] hyperParams) {
      ModelParametersBuilderFactory.ModelParametersBuilder<MP>
          paramsBuilder = _paramsBuilderFactory.get(params);
      for (int i = 0; i < _hyperParamNames.length; i++) {
        String paramName = _hyperParamNames[i];
        Object paramValue = hyperParams[i];

        if (paramName.equals("valid")) {  // change paramValue to key<Frame> for validation_frame
          paramName = "validation_frame";   // @#$, paramsSchema is still using validation_frame and training_frame
        }

        paramsBuilder.set(paramName, paramValue);
      }
      return paramsBuilder.build();
    }

    protected long computeMaxSizeOfHyperSpace() {
      long work = 1;
      for (Map.Entry<String, Object[]> p : _hyperParamsGrid.entrySet()) {
        if (p.getValue() != null) {
          work *= p.getValue().length;
        }
      }
      return work;
    }

    /** Given a list of indices for the hyperparameter values return an ordered Object[] of the actual values to represent single permutation. */
    protected Object[] permutation(int[] hidx, Object[] hypers) {
      for (int i = 0; i < hidx.length; i++) {
        hypers[i] = _hyperParamsGrid.get(_hyperParamNames[i])[hidx[i]];
      }
      return hypers;
    }

    protected int integerHash(int[] ar) {
      Integer[] hashMe = new Integer[ar.length];
      for (int i = 0; i < ar.length; i++)
        hashMe[i] = ar[i] * _hyperParamsGrid.get(_hyperParamNames[i]).length;
      return Arrays.deepHashCode(hashMe);
    }
  }

  /**
   * Hyperparameter space walker which visits each combination of hyperparameters in order.
   */
  public static class CartesianWalker<MP extends Model.Parameters>
          extends BaseWalker<MP, HyperSpaceSearchCriteria.CartesianSearchCriteria> {

    public CartesianWalker(MP params,
                           Map<String, Object[]> hyperParamsGrid,
                           ModelParametersBuilderFactory<MP> paramsBuilderFactory,
                           HyperSpaceSearchCriteria.CartesianSearchCriteria searchCriteria,
                           ArrayList<PermutationFilterFunction> filterFunctions) {
      super(params, hyperParamsGrid, paramsBuilderFactory, searchCriteria);

      _filterFunctions = filterFunctions;
    }
    
    public CartesianWalker(MP params,
                           Map<String, Object[]> hyperParamsGrid,
                           ModelParametersBuilderFactory<MP> paramsBuilderFactory,
                           HyperSpaceSearchCriteria.CartesianSearchCriteria searchCriteria) {
      this(params, hyperParamsGrid, paramsBuilderFactory, searchCriteria, null);
    }

    @Override
    public HyperSpaceIterator<MP> iterator() {

      return new HyperSpaceIterator<MP>() {
        /** Hyper params permutation.
         */
        private int[] _currentHyperparamIndices = null;

        @Override
        public MP nextModelParameters(Model previousModel) {
          _currentHyperparamIndices = _currentHyperparamIndices != null ? nextModelIndices(_currentHyperparamIndices) : new int[_hyperParamNames.length];
          if (_currentHyperparamIndices != null) {
            // Fill array of hyper-values
            Object[] permutation = permutation(_currentHyperparamIndices, new Object[_hyperParamNames.length]);

            if (_filterFunctions == null || !permutationIsSkipped(permutation, _hyperParamNames, _filterFunctions)) {
              // Get clone of parameters
              MP commonModelParams = (MP) _params.clone();
              // Fill model parameters
              MP params = getModelParams(commonModelParams, permutation);

              return params;
            } else {
              return hasNext(previousModel) ? nextModelParameters(previousModel) : null;
            }
          } else {
            throw new NoSuchElementException("No more elements to explore in hyper-space!");
          }
        }

        @Override
        public boolean hasNext(Model previousModel) {
          if (_currentHyperparamIndices == null) {
            return true;
          }
          int[] hyperparamIndices = _currentHyperparamIndices;
          for (int i = 0; i < hyperparamIndices.length; i++) {
            if (hyperparamIndices[i] + 1 < _hyperParamsGrid.get(_hyperParamNames[i]).length) {
              return true;
            }
          }
          return false;
        }

        @Override public void reset() {
          _currentHyperparamIndices = null;
          resetFilterFunctions(_filterFunctions);
        }

        @Override
        public double time_remaining_secs() { return Double.MAX_VALUE; }

        @Override
        public double max_runtime_secs() { return Double.MAX_VALUE; }

        public int max_models() { return _maxHyperSpaceSize > Integer.MAX_VALUE ? Integer.MAX_VALUE : (int)_maxHyperSpaceSize; }

        @Override
        public void modelFailed(Model failedModel) {
          // nada
        }

        @Override
        public Object[] getCurrentRawParameters() {
          Object[] hyperValues = new Object[_hyperParamNames.length];
          return permutation(_currentHyperparamIndices, hyperValues);
        }
      }; // anonymous HyperSpaceIterator class
    } // iterator()

    /**
     * Cartesian iteration over the hyper-parameter space, varying one hyperparameter at a
     * time. Mutates the indices that are passed in and returns them.  Returns NULL when
     * the entire space has been traversed.
     */
    private int[] nextModelIndices(int[] hyperparamIndices) {
      // Find the next parm to flip
      int i;
      for (i = 0; i < hyperparamIndices.length; i++) {
        if (hyperparamIndices[i] + 1 < _hyperParamsGrid.get(_hyperParamNames[i]).length) {
          break;
        }
      }
      if (i == hyperparamIndices.length) {
        return null; // All done, report null
      }
      // Flip indices
      for (int j = 0; j < i; j++) {
        hyperparamIndices[j] = 0;
      }
      hyperparamIndices[i]++;
      return hyperparamIndices;
    }
  } // class CartesianWalker

  /**
   * Hyperparameter space walker which visits random combinations of hyperparameters whose possible values are
   * given in explicit lists as they are with CartesianWalker.
   */
  public static class RandomDiscreteValueWalker<MP extends Model.Parameters>
      extends BaseWalker<MP, HyperSpaceSearchCriteria.RandomDiscreteValueSearchCriteria> {
    Random random;

    /** All visited hyper params permutations, including the current one. */
    private List<int[]> _visitedPermutations = new ArrayList<>();
    private Set<Integer> _visitedPermutationHashes = new LinkedHashSet<>(); // for fast dupe lookup

    /**
     * 
     * Keeps number of returned to the user permutations as not all visited permutations are considered to be worthy
     * for evaluation due to a {@code _filterFunctions}. 
     */
    private int _numberOfUsedPermutations = 0;

    public RandomDiscreteValueWalker(MP params,
                                     Map<String, Object[]> hyperParamsGrid,
                                     ModelParametersBuilderFactory<MP> paramsBuilderFactory,
                                     HyperSpaceSearchCriteria.RandomDiscreteValueSearchCriteria searchCriteria,
                                     ArrayList<PermutationFilterFunction> filterFunctions) {
      super(params, hyperParamsGrid, paramsBuilderFactory, searchCriteria);

      if (-1 == searchCriteria.seed())
        random = new Random();                       // true random
      else
        random = new Random(searchCriteria.seed()); // seeded repeatable pseudorandom
      
      _filterFunctions = filterFunctions;
    }
    
    public RandomDiscreteValueWalker(MP params,
                                     Map<String, Object[]> hyperParamsGrid,
                                     ModelParametersBuilderFactory<MP> paramsBuilderFactory,
                                     HyperSpaceSearchCriteria.RandomDiscreteValueSearchCriteria searchCriteria) {
      this(params, hyperParamsGrid, paramsBuilderFactory, searchCriteria, null);
      
    }

    /** Based on the last model, the given array of ScoringInfo, and our stopping criteria should we stop early? */
    @Override
    public boolean stopEarly(Model model, ScoringInfo[] sk) {
      return ScoreKeeper.stopEarly(ScoringInfo.scoreKeepers(sk),
                                   search_criteria().stopping_rounds(),
                                    ScoreKeeper.ProblemType.forSupervised(model._output.isClassifier()),
                                   search_criteria().stopping_metric(),
                                   search_criteria().stopping_tolerance(), "grid's best", true);
    }

    @Override
    public HyperSpaceIterator<MP> iterator() {
      return new HyperSpaceIterator<MP>() {
        /** Current hyper params permutation. */
        private int[] _currentHyperparamIndices = null;

        /** One-based count of the permutations we've visited, primarily used as an index into _visitedHyperparamIndices. */
        private int _currentPermutationNum = 0;

        /** Start time of this grid */
        private long _start_time = System.currentTimeMillis();

        // TODO: override into a common subclass:
        @Override
        public MP nextModelParameters(Model previousModel) {
          // NOTE: nextModel checks _visitedHyperparamIndices and does not return a duplicate set of indices.
          // NOTE: in RandomDiscreteValueWalker nextModelIndices() returns a new array each time, rather than
          // mutating the last one.

          _currentHyperparamIndices = nextModelIndices();

          if (_currentHyperparamIndices != null) {
            _visitedPermutations.add(_currentHyperparamIndices);
            _visitedPermutationHashes.add(integerHash(_currentHyperparamIndices));
            _currentPermutationNum++; // NOTE: 1-based counting

            // Fill array of hyper-values
            Object[] permutation = permutation(_currentHyperparamIndices, new Object[_hyperParamNames.length]);
             
            if (_filterFunctions == null || !permutationIsSkipped(permutation, _hyperParamNames, _filterFunctions)) {

              _numberOfUsedPermutations++;
              // Get clone of parameters
              MP commonModelParams = (MP) _params.clone();
              // Fill model parameters
              MP params = getModelParams(commonModelParams, permutation);

              // add max_runtime_secs in search criteria into params if applicable
              if (_search_criteria != null && _search_criteria.strategy() == HyperSpaceSearchCriteria.Strategy.RandomDiscrete) {
                // ToDo: model seed setting will be different for parallel model building.
                // ToDo: This implementation only works for sequential model building.
                if (_set_model_seed_from_search_seed) {
                  // set model seed = search_criteria.seed+(0, 1, 2,..., model number)
                  params._seed = _search_criteria.seed() + (model_number++);
                }

                // set max_runtime_secs
                double timeleft = this.time_remaining_secs();
                if (timeleft > 0) {
                  if (params._max_runtime_secs > 0) {
                    params._max_runtime_secs = min(params._max_runtime_secs, timeleft);
                  } else {
                    params._max_runtime_secs = timeleft;
                  }
                }
              }
              return params;
            } else {
              
              int nextPermutationNum = _currentPermutationNum + 1;
              if(hasNext(previousModel) && (max_models() == 0 || max_models() > 0 && nextPermutationNum < max_models())) {
                return nextModelParameters(previousModel);
              } else {
                return null;
              }
            }
          } else {
            throw new NoSuchElementException("No more elements to explore in hyper-space!");
          }
        }

        @Override
        public boolean hasNext(Model previousModel) {
          // Note: we compare _currentPermutationNum to max_models, because it counts successfully created models, but
          // we compare _visitedPermutationHashes.size() to _maxHyperSpaceSize because we want to stop when we have attempted each combo.
          //
          // _currentPermutationNum is 1-based
          return (_visitedPermutationHashes.size() < _maxHyperSpaceSize &&
                  (search_criteria().max_models() == 0 || _numberOfUsedPermutations < search_criteria().max_models())
          );
        }

        @Override
        public void reset() {
          _start_time = System.currentTimeMillis();
          _currentPermutationNum = 0;
          _currentHyperparamIndices = null;
          _visitedPermutations.clear();
          _visitedPermutationHashes.clear();
          resetFilterFunctions(_filterFunctions);
          _numberOfUsedPermutations = 0;
        }

        public double max_runtime_secs() {
          return search_criteria().max_runtime_secs();
        }

        public int max_models() {
          return search_criteria().max_models();
        }

        @Override
        public double time_remaining_secs() {
          return search_criteria().max_runtime_secs() - (System.currentTimeMillis() - _start_time) / 1000.0;
        }

        @Override
        public void modelFailed(Model failedModel) {
          // Leave _visitedPermutations, _visitedPermutationHashes and _currentHyperparamIndices alone
          // so we don't revisit bad parameters. Note that if a model build fails for other reasons we
          // won't retry.
          _currentPermutationNum--;
          _numberOfUsedPermutations--;
        }

        @Override
        public Object[] getCurrentRawParameters() {
          Object[] hyperValues = new Object[_hyperParamNames.length];
          return permutation(_currentHyperparamIndices, hyperValues);
        }
      }; // anonymous HyperSpaceIterator class
    } // iterator()

    /**
     * Random iteration over the hyper-parameter space.  Does not repeat
     * previously-visited combinations.  Returns NULL when we've hit the stopping
     * criteria.
     */
    private int[] nextModelIndices() {
      int[] hyperparamIndices =  new int[_hyperParamNames.length];

      do {
        // generate random indices
        for (int i = 0; i < _hyperParamNames.length; i++) {
          hyperparamIndices[i] = random.nextInt(_hyperParamsGrid.get(_hyperParamNames[i]).length);
        }
        // check for aliases and loop if we've visited this combo before
      } while (_visitedPermutationHashes.contains(integerHash(hyperparamIndices)));

      return hyperparamIndices;
    } // nextModel

  } // RandomDiscreteValueWalker
}
