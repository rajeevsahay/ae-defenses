import numpy as np
import tensorflow as tf
from abc import ABCMeta
import collections
import warnings
from six.moves import xrange

from cleverhans import utils
from cleverhans.model import Model, CallableModelWrapper
from cleverhans_compat import reduce_sum, reduce_mean
from cleverhans_compat import reduce_max
from cleverhans_compat import softmax_cross_entropy_with_logits


_logger = utils.create_logger("cleverhans.attacks")
tf_dtype = tf.as_dtype('float32')
class Attack(object):
  """
  Abstract base class for all attack classes.
  """
  __metaclass__ = ABCMeta

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    """
    :param model: An instance of the cleverhans.model.Model class.
    :param sess: The (possibly optional) tf.Session to run graphs in.
    :param dtypestr: Floating point precision to use (change to float64
                     to avoid numerical instabilities).
    :param back: (deprecated and will be removed on or after 2019-03-26).
                 The backend to use. Currently 'tf' is the only option.
    """
    if 'back' in kwargs:
      if kwargs['back'] == 'tf':
        warnings.warn("Argument back to attack constructors is not needed"
                      " anymore and will be removed on or after 2019-03-26."
                      " All attacks are implemented using TensorFlow.")
      else:
        raise ValueError("Backend argument must be 'tf' and is now deprecated"
                         "It will be removed on or after 2019-03-26.")

    self.tf_dtype = tf.as_dtype(dtypestr)
    self.np_dtype = np.dtype(dtypestr)

    if sess is not None and not isinstance(sess, tf.Session):
      raise TypeError("sess is not an instance of tf.Session")

    from cleverhans import attacks_tf
    attacks_tf.np_dtype = self.np_dtype
    attacks_tf.tf_dtype = self.tf_dtype

    if not isinstance(model, Model):
      raise TypeError("The model argument should be an instance of"
                      " the cleverhans.model.Model class.")

    # Prepare attributes
    self.model = model
    self.sess = sess
    self.dtypestr = dtypestr

    # We are going to keep track of old graphs and cache them.
    self.graphs = {}

    # When calling generate_np, arguments in the following set should be
    # fed into the graph, as they are not structural items that require
    # generating a new graph.
    # This dict should map names of arguments to the types they should
    # have.
    # (Usually, the target class will be a feedable keyword argument.)
    self.feedable_kwargs = tuple()

    # When calling generate_np, arguments in the following set should NOT
    # be fed into the graph, as they ARE structural items that require
    # generating a new graph.
    # This list should contain the names of the structural arguments.
    self.structural_kwargs = []

  def generate(self, x, **kwargs):
    """
    Generate the attack's symbolic graph for adversarial examples. This
    method should be overriden in any child class that implements an
    attack that is expressable symbolically. Otherwise, it will wrap the
    numerical implementation as a symbolic operator.
    :param x: The model's symbolic inputs.
    :param **kwargs: optional parameters used by child classes.
      Each child class defines additional parameters as needed.
      Child classes that use the following concepts should use the following
      names:
        clip_min: minimum feature value
        clip_max: maximum feature value
        eps: size of norm constraint on adversarial perturbation
        ord: order of norm constraint
        nb_iter: number of iterations
        eps_iter: size of norm constraint on iteration
        y_target: if specified, the attack is targeted.
        y: Do not specify if y_target is specified.
           If specified, the attack is untargeted, aims to make the output
           class not be y.
           If neither y_target nor y is specified, y is inferred by having
           the model classify the input.
      For other concepts, it's generally a good idea to read other classes
      and check for name consistency.
    :return: A symbolic representation of the adversarial examples.
    """

    error = "Sub-classes must implement generate."
    raise NotImplementedError(error)
    # Include an unused return so pylint understands the method signature
    return x

  def construct_graph(self, fixed, feedable, x_val, hash_key):
    """
    Construct the graph required to run the attack through generate_np.
    :param fixed: Structural elements that require defining a new graph.
    :param feedable: Arguments that can be fed to the same graph when
                     they take different values.
    :param x_val: symbolic adversarial example
    :param hash_key: the key used to store this graph in our cache
    """
    # try our very best to create a TF placeholder for each of the
    # feedable keyword arguments, and check the types are one of
    # the allowed types
    class_name = str(self.__class__).split(".")[-1][:-2]
    _logger.info("Constructing new graph for attack " + class_name)

    # remove the None arguments, they are just left blank
    for k in list(feedable.keys()):
      if feedable[k] is None:
        del feedable[k]



    # process all of the rest and create placeholders for them
    new_kwargs = dict(x for x in fixed.items())
    for name, value in feedable.items():
      given_type = value.dtype
      if isinstance(value, np.ndarray):
        if value.ndim == 0:
          # This is pretty clearly not a batch of data
          new_kwargs[name] = tf.placeholder(given_type, shape=[], name=name)
        else:
          # Assume that this is a batch of data, make the first axis variable
          # in size
          new_shape = [None] + list(value.shape[1:])
          new_kwargs[name] = tf.placeholder(given_type, new_shape, name=name)
      elif isinstance(value, utils.known_number_types):
        new_kwargs[name] = tf.placeholder(given_type, shape=[], name=name)
      else:
        raise ValueError("Could not identify type of argument " +
                         name + ": " + str(value))

    # x is a special placeholder we always want to have
    x_shape = [None] + list(x_val.shape)[1:]
    x = tf.placeholder(self.tf_dtype, shape=x_shape)

    # now we generate the graph that we want
    x_adv = self.generate(x, **new_kwargs)

    self.graphs[hash_key] = (x, new_kwargs, x_adv)

    if len(self.graphs) >= 10:
      warnings.warn("Calling generate_np() with multiple different "
                    "structural parameters is inefficient and should"
                    " be avoided. Calling generate() is preferred.")

  def generate_np(self, x_val, **kwargs):
    """
    Generate adversarial examples and return them as a NumPy array.
    Sub-classes *should not* implement this method unless they must
    perform special handling of arguments.
    :param x_val: A NumPy array with the original inputs.
    :param **kwargs: optional parameters used by child classes.
    :return: A NumPy array holding the adversarial examples.
    """

    if self.sess is None:
      raise ValueError("Cannot use `generate_np` when no `sess` was"
                       " provided")

    packed = self.construct_variables(kwargs)
    fixed, feedable, _, hash_key = packed

    if hash_key not in self.graphs:
      self.construct_graph(fixed, feedable, x_val, hash_key)
    else:
      # remove the None arguments, they are just left blank
      for k in list(feedable.keys()):
        if feedable[k] is None:
          del feedable[k]

    x, new_kwargs, x_adv = self.graphs[hash_key]

    feed_dict = {x: x_val}

    for name in feedable:
      feed_dict[new_kwargs[name]] = feedable[name]

    return self.sess.run(x_adv, feed_dict)

  def construct_variables(self, kwargs):
    """
    Construct the inputs to the attack graph to be used by generate_np.
    :param kwargs: Keyword arguments to generate_np.
    :return:
      Structural arguments
      Feedable arguments
      Output of `arg_type` describing feedable arguments
      A unique key
    """
    if isinstance(self.feedable_kwargs, dict):
      warnings.warn("Using a dict for `feedable_kwargs is deprecated."
                    "Switch to using a tuple."
                    "It is not longer necessary to specify the types "
                    "of the arguments---we build a different graph "
                    "for each received type."
                    "Using a dict may become an error on or after "
                    "2019-04-18.")
      feedable_names = tuple(sorted(self.feedable_kwargs.keys()))
    else:
      feedable_names = self.feedable_kwargs
      if not isinstance(feedable_names, tuple):
        raise TypeError("Attack.feedable_kwargs should be a tuple, but "
                        "for subclass " + str(type(self)) + " it is "
                        + str(self.feedable_kwargs) + " of type "
                        + str(type(self.feedable_kwargs)))

    # the set of arguments that are structural properties of the attack
    # if these arguments are different, we must construct a new graph
    fixed = dict(
        (k, v) for k, v in kwargs.items() if k in self.structural_kwargs)

    # the set of arguments that are passed as placeholders to the graph
    # on each call, and can change without constructing a new graph
    feedable = {k: v for k, v in kwargs.items() if k in feedable_names}
    for k in feedable:
      if isinstance(feedable[k], (float, int)):
        feedable[k] = np.array(feedable[k])

    for key in kwargs:
      if key not in fixed and key not in feedable:
        raise ValueError(str(type(self)) + ": Undeclared argument: " + key)

    feed_arg_type = arg_type(feedable_names, feedable)

    if not all(isinstance(value, collections.Hashable)
               for value in fixed.values()):
      # we have received a fixed value that isn't hashable
      # this means we can't cache this graph for later use,
      # and it will have to be discarded later
      hash_key = None
    else:
      # create a unique key for this set of fixed paramaters
      hash_key = tuple(sorted(fixed.items())) + tuple([feed_arg_type])

    return fixed, feedable, feed_arg_type, hash_key

  def get_or_guess_labels(self, x, kwargs):
    """
    Get the label to use in generating an adversarial example for x.
    The kwargs are fed directly from the kwargs of the attack.
    If 'y' is in kwargs, then assume it's an untargeted attack and
    use that as the label.
    If 'y_target' is in kwargs and is not none, then assume it's a
    targeted attack and use that as the label.
    Otherwise, use the model's prediction as the label and perform an
    untargeted attack.
    """
    if 'y' in kwargs and 'y_target' in kwargs:
      raise ValueError("Can not set both 'y' and 'y_target'.")
    elif 'y' in kwargs:
      labels = kwargs['y']
    elif 'y_target' in kwargs and kwargs['y_target'] is not None:
      labels = kwargs['y_target']
    else:
      preds = self.model.get_probs(x)
      preds_max = reduce_max(preds, 1, keepdims=True)
      original_predictions = tf.to_float(tf.equal(preds, preds_max))
      labels = tf.stop_gradient(original_predictions)
      del preds
    if isinstance(labels, np.ndarray):
      nb_classes = labels.shape[1]
    else:
      nb_classes = labels.get_shape().as_list()[1]
    return labels, nb_classes

  def parse_params(self, params=None):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.
    :param params: a dictionary of attack-specific parameters
    :return: True when parsing was successful
    """

    if params is not None:
      warnings.warn("`params` is unused and will be removed "
                    " on or after 2019-04-26.")
    return True



class FastGradientMethod(Attack):
  """
  This attack was originally implemented by Goodfellow et al. (2015) with the
  infinity norm (and is known as the "Fast Gradient Sign Method"). This
  implementation extends the attack to other norms, and is therefore called
  the Fast Gradient Method.
  Paper link: https://arxiv.org/abs/1412.6572
  :param model: cleverhans.model.Model
  :param sess: optional tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor
  """

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    """
    Create a FastGradientMethod instance.
    Note: the model parameter should be an instance of the
    cleverhans.model.Model abstraction provided by CleverHans.
    """

    super(FastGradientMethod, self).__init__(model, sess, dtypestr, **kwargs)
    self.feedable_kwargs = ('eps', 'y', 'y_target', 'clip_min', 'clip_max')
    self.structural_kwargs = ['ord', 'sanity_checks']

  def generate(self, x, **kwargs):
    """
    Returns the graph for Fast Gradient Method adversarial examples.
    :param x: The model's symbolic inputs.
    :param kwargs: See `parse_params`
    """
    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    labels, _nb_classes = self.get_or_guess_labels(x, kwargs)

    return fgm(
        x,
        self.model.get_logits(x),
        y=labels,
        eps=self.eps,
        ord=self.ord,
        clip_min=self.clip_min,
        clip_max=self.clip_max,
        targeted=(self.y_target is not None),
        sanity_checks=self.sanity_checks)

  def parse_params(self,
                   eps=0.3,
                   ord=np.inf,
                   y=None,
                   y_target=None,
                   clip_min=None,
                   clip_max=None,
                   sanity_checks=True,
                   **kwargs):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.
    Attack-specific parameters:
    :param eps: (optional float) attack step size (input variation)
    :param ord: (optional) Order of the norm (mimics NumPy).
                Possible values: np.inf, 1 or 2.
    :param y: (optional) A tensor with the true labels. Only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param y_target: (optional) A tensor with the labels to target. Leave
                     y_target=None if y is also set. Labels should be
                     one-hot-encoded.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    :param sanity_checks: bool, if True, include asserts
      (Turn them off to use less runtime / memory or for unit tests that
      intentionally pass strange input)
    """
    # Save attack-specific parameters

    self.eps = eps
    self.ord = ord
    self.y = y
    self.y_target = y_target
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.sanity_checks = sanity_checks

    if self.y is not None and self.y_target is not None:
      raise ValueError("Must not set both y and y_target")
    # Check if order of the norm is acceptable given current implementation
    if self.ord not in [np.inf, int(1), int(2)]:
      raise ValueError("Norm order must be either np.inf, 1, or 2.")

    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")

    return True


def fgm(x,
        logits,
        y=None,
        eps=0.3,
        ord=np.inf,
        clip_min=None,
        clip_max=None,
        targeted=False,
        sanity_checks=True):
  """
  TensorFlow implementation of the Fast Gradient Method.
  :param x: the input placeholder
  :param logits: output of model.get_logits
  :param y: (optional) A placeholder for the true labels. If targeted
            is true, then provide the target label. Otherwise, only provide
            this parameter if you'd like to use true labels when crafting
            adversarial samples. Otherwise, model predictions are used as
            labels to avoid the "label leaking" effect (explained in this
            paper: https://arxiv.org/abs/1611.01236). Default is None.
            Labels should be one-hot-encoded.
  :param eps: the epsilon (input variation parameter)
  :param ord: (optional) Order of the norm (mimics NumPy).
              Possible values: np.inf, 1 or 2.
  :param clip_min: Minimum float value for adversarial example components
  :param clip_max: Maximum float value for adversarial example components
  :param targeted: Is the attack targeted or untargeted? Untargeted, the
                   default, will try to make the label incorrect. Targeted
                   will instead try to move in the direction of being more
                   like y.
  :return: a tensor for the adversarial example
  """

  asserts = []

  # If a data range was specified, check that the input was in that range
  #if clip_min is not None:
  #  asserts.append(utils_tf.assert_greater_equal(
  #      x, tf.cast(clip_min, x.dtype)))

  #if clip_max is not None:
  #  asserts.append(utils_tf.assert_less_equal(x, tf.cast(clip_max, x.dtype)))

  # Make sure the caller has not passed probs by accident
  assert logits.op.type != 'Softmax'

  if y is None:
    # Using model predictions as ground truth to avoid label leaking
    preds_max = reduce_max(logits, 1, keepdims=True)
    y = tf.to_float(tf.equal(logits, preds_max))
    y = tf.stop_gradient(y)
  y = y / reduce_sum(y, 1, keepdims=True)

  # Compute loss
  loss = softmax_cross_entropy_with_logits(labels=y, logits=logits)
  if targeted:
    loss = -loss

  # Define gradient of loss wrt input
  grad, = tf.gradients(loss, x)

  optimal_perturbation = optimize_linear(grad, eps, ord)

  # Add perturbation to original example to obtain adversarial example
  adv_x = x + optimal_perturbation

  # If clipping is needed, reset all values outside of [clip_min, clip_max]
  if (clip_min is not None) or (clip_max is not None):
    # We don't currently support one-sided clipping
    assert clip_min is not None and clip_max is not None
    adv_x = clip_by_value(adv_x, clip_min, clip_max)

  if sanity_checks:
    with tf.control_dependencies(asserts):
      adv_x = tf.identity(adv_x)

  return adv_x


def optimize_linear(grad, eps, ord=np.inf):
  """
  Solves for the optimal input to a linear function under a norm constraint.
  Optimal_perturbation = argmax_{eta, ||eta||_{ord} < eps} dot(eta, grad)
  :param grad: tf tensor containing a batch of gradients
  :param eps: float scalar specifying size of constraint region
  :param ord: int specifying order of norm
  :returns:
    tf tensor containing optimal perturbation
  """

  red_ind = list(xrange(1, len(grad.get_shape())))
  avoid_zero_div = 1e-12
  if ord == np.inf:
    # Take sign of gradient
    optimal_perturbation = tf.sign(grad)
    # The following line should not change the numerical results.
    # It applies only because `optimal_perturbation` is the output of
    # a `sign` op, which has zero derivative anyway.
    # It should not be applied for the other norms, where the
    # perturbation has a non-zero derivative.
    optimal_perturbation = tf.stop_gradient(optimal_perturbation)
  elif ord == 1:
    abs_grad = tf.abs(grad)
    sign = tf.sign(grad)
    max_abs_grad = tf.reduce_max(abs_grad, red_ind, keepdims=True)
    tied_for_max = tf.to_float(tf.equal(abs_grad, max_abs_grad))
    num_ties = tf.reduce_sum(tied_for_max, red_ind, keepdims=True)
    optimal_perturbation = sign * tied_for_max / num_ties
  elif ord == 2:
    square = tf.maximum(avoid_zero_div,
                        reduce_sum(tf.square(grad),
                                   reduction_indices=red_ind,
                                   keepdims=True))
    optimal_perturbation = grad / tf.sqrt(square)
  else:
    raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                              "currently implemented.")

  # Scale perturbation to be the solution for the norm=eps rather than
  # norm=1 problem
  scaled_perturbation = mul(eps, optimal_perturbation)
  return scaled_perturbation

def arg_type(arg_names, kwargs):
  """
  Returns a hashable summary of the types of arg_names within kwargs.
  :param arg_names: tuple containing names of relevant arguments
  :param kwargs: dict mapping string argument names to values.
    These must be values for which we can create a tf placeholder.
    Currently supported: numpy darray or something that can ducktype it
  returns:
    API contract is to return a hashable object describing all
    structural consequences of argument values that can otherwise
    be fed into a graph of fixed structure.
    Currently this is implemented as a tuple of tuples that track:
      - whether each argument was passed
      - whether each argument was passed and not None
      - the dtype of each argument
    Callers shouldn't rely on the exact structure of this object,
    just its hashability and one-to-one mapping between graph structures.
  """
  assert isinstance(arg_names, tuple)
  passed = tuple(name in kwargs for name in arg_names)
  passed_and_not_none = []
  for name in arg_names:
    if name in kwargs:
      passed_and_not_none.append(kwargs[name] is not None)
    else:
      passed_and_not_none.append(False)
  passed_and_not_none = tuple(passed_and_not_none)
  dtypes = []
  for name in arg_names:
    if name not in kwargs:
      dtypes.append(None)
      continue
    value = kwargs[name]
    if value is None:
      dtypes.append(None)
      continue
    assert hasattr(value, 'dtype'), type(value)
    dtype = value.dtype
    if not isinstance(dtype, np.dtype):
      dtype = dtype.as_np_dtype
    assert isinstance(dtype, np.dtype)
    dtypes.append(dtype)
  dtypes = tuple(dtypes)
  return (passed, passed_and_not_none, dtypes)

def mul(a, b):
  """
  A wrapper around tf multiplication that does more automatic casting of
  the input.
  """
  def multiply(a, b):
    return a * b
  return op_with_scalar_cast(a, b, multiply)

def op_with_scalar_cast(a, b, f):
  """
  Builds the graph to compute f(a, b).
  If only one of the two arguments is a scalar and the operation would
  cause a type error without casting, casts the scalar to match the
  tensor.
  :param a: a tf-compatible array or scalar
  :param b: a tf-compatible array or scalar
  """

  try:
    return f(a, b)
  except (TypeError, ValueError):
    pass

  def is_scalar(x):
    if hasattr(x, "get_shape"):
      shape = x.get_shape()
      return shape.ndims == 0
    if hasattr(x, "ndim"):
      return x.ndim == 0
    assert isinstance(x, (int, float))
    return True

  a_scalar = is_scalar(a)
  b_scalar = is_scalar(b)

  if a_scalar and b_scalar:
    raise TypeError("Trying to apply " + str(f) + " with mixed types")

  if a_scalar and not b_scalar:
    a = tf.cast(a, b.dtype)

  if b_scalar and not a_scalar:
    b = tf.cast(b, a.dtype)

  return f(a, b)


def clip_by_value(t, clip_value_min, clip_value_max, name=None):
  """
  A wrapper for clip_by_value that casts the clipping range if needed.
  """
  def cast_clip(clip):
    if t.dtype in (tf.float32, tf.float64):
      if hasattr(clip, 'dtype'):
        if clip.dtype != t.dtype:
          return tf.cast(clip, t.dtype)
    return clip

  clip_value_min = cast_clip(clip_value_min)
  clip_value_max = cast_clip(clip_value_max)

  return tf.clip_by_value(t, clip_value_min, clip_value_max, name)
