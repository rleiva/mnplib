# Model serializer examples

This document compares the strings emitted by the current mnplib serializers with complete illustrative programs in the proposed restricted Python grammar. It covers every registered scikit-learn estimator type and every forecasting family, plus multi-output regression, neural-network activations, preprocessing, and both supported structural state-space specifications.

These examples are a design proposal, not a change to the library serializers.

## Scope and conventions

- The current strings below were obtained directly from the repository serializers and are reproduced verbatim.
- Small controlled fixtures keep the coefficients readable. The scikit-learn estimators were fitted to establish their structure, then selected fitted parameter arrays were set to the stated simple values. Statistical time-series results were filtered with explicit parameters and known initial states; these are not claims about maximum-likelihood parameter estimates.
- Every proposed code block is complete for its stated example. It has no imports, external model objects, omitted helper bodies, or mathematical library calls.
- Classification outputs are zero-based class indices, not probabilities. A class-label mapping remains separate metadata.
- Tabular inputs use original feature coordinates. The examples use feature zero, so they do not exercise nontrivial subset remapping.
- For lagged predictors, x[0] means the latest observed value, x[1] the preceding value. Lag construction is part of the declared input convention.
- Stateful programs expose initial_state(), step(observation, state), and forecast(state). step returns [prediction_before_observation, next_state]; forecast returns [prediction, next_state] without observing a new value.
- The examples preserve meaningful fitted numerical constants. Reporting names, fit statistics, and descriptive headers are not part of the proposed executable program.
- All program text, including activation helpers and initial-state literals, would count toward model description length. No external helper library is assumed.

## Contents

1. [LinearRegression](#example-linear)
2. [LinearRegression with two outputs](#example-multi)
3. [LogisticRegression](#example-logistic)
4. [LinearSVC](#example-svc)
5. [LinearSVR](#example-svr)
6. [DecisionTreeClassifier](#example-treec)
7. [DecisionTreeRegressor](#example-treer)
8. [GaussianNB](#example-nb)
9. [MLPClassifier with ReLU](#example-mlpc)
10. [MLPRegressor with ReLU](#example-mlpr)
11. [MLPRegressor with StandardScaler](#example-scaled)
12. [MLPRegressor with logistic activation](#example-mlpl)
13. [MLPRegressor with tanh activation](#example-mlpt)
14. [Autoregressive linear forecast](#example-ar)
15. [Moving-average forecast](#example-ma)
16. [Finite-window exponential smoothing](#example-es)
17. [ARIMA with order (1, 0, 0)](#example-arima)
18. [State-space local level](#example-level)
19. [State-space local linear trend](#example-trend)

<a id="example-linear"></a>

## 1 LinearRegression

One input, coefficient 2, and intercept 1. Ordinary assignment replaces augmented assignment; parentheses follow the proposed grammar.

### Current output

```text
def predict(x):
 y = 1.00e+00
 y += 2.00e+00*x[0]
 return y
```

### Proposed output

```python
def predict(x):
    y = (1.0 + (2.0 * x[0]))
    return y
```

<a id="example-multi"></a>

## 2 LinearRegression with two outputs

The two outputs are 1 + 2*x[0] and 3 - x[0]. The proposed list has the required comma. The current string is not executable Python.

### Current output

```text
def predict(x):
 y_0 = 1.00e+00
 y_0 += 2.00e+00*x[0]
 y_1 = 3.00e+00
 y_1 -= 1.00e+00*x[0]
 return [y_0 y_1]
```

### Proposed output

```python
def predict(x):
    y0 = (1.0 + (2.0 * x[0]))
    y1 = (3.0 - x[0])
    return [y0, y1]
```

<a id="example-logistic"></a>

## 3 LogisticRegression

Binary classes are represented by indices 0 and 1. The fitted logit is -1 + 2*x[0]. Only its sign is needed for class prediction, so no sigmoid is evaluated. This does not implement predict_proba().

### Current output

```text
def predict(x):
 z=-1.00e+00+2.00e+00*x[0]
 if z>0:
  return 1
 return 0
```

### Proposed output

```python
def predict(x):
    z = (-1.0 + (2.0 * x[0]))
    if z > 0.0:
        return 1
    return 0
```

<a id="example-svc"></a>

## 4 LinearSVC

A binary linear support-vector classifier uses the same sign test as this logistic classifier. Equal discriminant functions should receive the same executable description, regardless of the training algorithm.

### Current output

```text
def predict(x):
 z=-1.00e+00+2.00e+00*x[0]
 if z>0:
  return 1
 return 0
```

### Proposed output

```python
def predict(x):
    z = (-1.0 + (2.0 * x[0]))
    if z > 0.0:
        return 1
    return 0
```

<a id="example-svr"></a>

## 5 LinearSVR

A linear support-vector regressor is an affine predictor. Its proposed description intentionally matches LinearRegression when their fitted coefficients match.

### Current output

```text
def predict(x):
 y=1.00e+00+2.00e+00*x[0]
 return y
```

### Proposed output

```python
def predict(x):
    y = (1.0 + (2.0 * x[0]))
    return y
```

<a id="example-treec"></a>

## 6 DecisionTreeClassifier

The root splits at zero and the leaves represent class indices 0 and 1. The function and indexed input make every reference explicit.

### Current output

```text
if X0<=0:
 return C0
else:
 return C1
```

### Proposed output

```python
def predict(x):
    if x[0] <= 0.0:
        return 0
    else:
        return 1
```

<a id="example-treer"></a>

## 7 DecisionTreeRegressor

The root splits at zero and returns 1 or 3. No estimator object is required to execute the proposed function.

### Current output

```text
if X0<=0:
 return 1.00e+00
else:
 return 3.00e+00
```

### Proposed output

```python
def predict(x):
    if x[0] <= 0.0:
        return 1.0
    else:
        return 3.0
```

<a id="example-nb"></a>

## 8 GaussianNB

Two equally likely classes have means -1.5 and 1.5 and variance 0.25. The constant -0.9189385332046727 is the precomputed sum of the log prior and Gaussian normalization term. Multiplication replaces squaring. No logarithm or exponential is evaluated during prediction.

### Current output

```text
def predict(x):
 s0=-6.93e-01-2.26e-01-((x[0]--1.50e+00)**2)/5.00e-01
 best=0
 best_s=s0
 s1=-6.93e-01-2.26e-01-((x[0]-1.50e+00)**2)/5.00e-01
 if s1>best_s:
  best=1
  best_s=s1
 return best
```

### Proposed output

```python
def predict(x):
    d0 = (x[0] - -1.5)
    s0 = (-0.9189385332046727 - ((d0 * d0) / 0.5))
    d1 = (x[0] - 1.5)
    s1 = (-0.9189385332046727 - ((d1 * d1) / 0.5))
    if s1 > s0:
        return 1
    return 0
```

<a id="example-mlpc"></a>

## 9 MLPClassifier with ReLU

One input, one ReLU hidden unit, and one binary output. The hidden unit is max(0, 2*x[0] - 1); the final logit is 3*h - 1. Its sign determines the class. The same list-and-loop template can represent larger networks.

### Current output

```text
def predict(x):
 W=[[[2.00e+00]],[[3.00e+00]]]
 B=[[-1.00e+00],[-1.00e+00]]
 a=[x[0]]
 for l in range(len(W)):
  h=[]
  for j in range(len(B[l])):
   z=B[l][j]
   for i in range(len(a)):
    z+=a[i]*W[l][i][j]
   if l<len(W)-1:
    if z>0:
     h.append(z)
    else:
     h.append(0)
   else:
    h.append(z)
  a=h
 if a[0]>0:
  return 1
 return 0
```

### Proposed output

```python
def predict(x):
    W = [[[2.0]], [[3.0]]]
    B = [[-1.0], [-1.0]]
    a = [x[0]]
    for l in range(len(W)):
        h = []
        for j in range(len(B[l])):
            z = B[l][j]
            for i in range(len(a)):
                z = (z + (a[i] * W[l][i][j]))
            if l < (len(W) - 1):
                if z > 0.0:
                    h.append(z)
                else:
                    h.append(0.0)
            else:
                h.append(z)
        a = h
    if a[0] > 0.0:
        return 1
    return 0
```

<a id="example-mlpr"></a>

## 10 MLPRegressor with ReLU

The hidden unit is max(0, 2*x[0] - 1); the regression output is 3*h + 0.5. The proposed program uses ordinary assignments and explicitly parenthesized arithmetic.

### Current output

```text
def predict(x):
 W=[[[2.00e+00]],[[3.00e+00]]]
 B=[[-1.00e+00],[5.00e-01]]
 a=[x[0]]
 for l in range(len(W)):
  h=[]
  for j in range(len(B[l])):
   z=B[l][j]
   for i in range(len(a)):
    z+=a[i]*W[l][i][j]
   if l<len(W)-1:
    if z>0:
     h.append(z)
    else:
     h.append(0)
   else:
    h.append(z)
  a=h
 return a[0]
```

### Proposed output

```python
def predict(x):
    W = [[[2.0]], [[3.0]]]
    B = [[-1.0], [0.5]]
    a = [x[0]]
    for l in range(len(W)):
        h = []
        for j in range(len(B[l])):
            z = B[l][j]
            for i in range(len(a)):
                z = (z + (a[i] * W[l][i][j]))
            if l < (len(W) - 1):
                if z > 0.0:
                    h.append(z)
                else:
                    h.append(0.0)
            else:
                h.append(z)
        a = h
    return a[0]
```

<a id="example-scaled"></a>

## 11 MLPRegressor with StandardScaler

The model input is standardized with mean 1 and scale 2 before the ReLU network. The proposed function accepts raw input and includes the scaling operation in its measured code, without modifying the caller's list.

### Current output

```text
P StandardScaler
 X0 = (X0-1.00e+00)/2.00e+00
def predict(x):
 W=[[[2.00e+00]],[[3.00e+00]]]
 B=[[-1.00e+00],[5.00e-01]]
 a=[x[0]]
 for l in range(len(W)):
  h=[]
  for j in range(len(B[l])):
   z=B[l][j]
   for i in range(len(a)):
    z+=a[i]*W[l][i][j]
   if l<len(W)-1:
    if z>0:
     h.append(z)
    else:
     h.append(0)
   else:
    h.append(z)
  a=h
 return a[0]
```

### Proposed output

```python
def predict(x):
    W = [[[2.0]], [[3.0]]]
    B = [[-1.0], [0.5]]
    a = [((x[0] - 1.0) / 2.0)]
    for l in range(len(W)):
        h = []
        for j in range(len(B[l])):
            z = B[l][j]
            for i in range(len(a)):
                z = (z + (a[i] * W[l][i][j]))
            if l < (len(W) - 1):
                if z > 0.0:
                    h.append(z)
                else:
                    h.append(0.0)
            else:
                h.append(z)
        a = h
    return a[0]
```

<a id="example-mlpl"></a>

## 12 MLPRegressor with logistic activation

The hidden unit is logistic(2*x[0] - 1), followed by 3*h + 0.5. The complete exponential helper is included in the proposed model string. It uses a 20-term Taylor expansion after division by 256, followed by eight squarings. This is an illustrative numerical approximation, checked at 801 arguments in [-40, 40], not a production implementation for arbitrary real inputs. No clipping of the logistic activation is implied. An implementation must define and test its numerical domain and error policy before accepting this activation.

### Current output

```text
def predict(x):
 E=2.718281828459045
 W=[[[2.00e+00]],[[3.00e+00]]]
 B=[[-1.00e+00],[5.00e-01]]
 a=[x[0]]
 for l in range(len(W)):
  h=[]
  for j in range(len(B[l])):
   z=B[l][j]
   for i in range(len(a)):
    z+=a[i]*W[l][i][j]
   if l<len(W)-1:
    if z>=0:
     h.append(1/(1+E**(-z)))
    else:
     e=E**z
     h.append(e/(1+e))
   else:
    h.append(z)
  a=h
 return a[0]
```

### Proposed output

```python
def exponential(z):
    t = (z / 256.0)
    total = 1.0
    term = 1.0
    for i in range(20):
        term = ((term * t) / (i + 1))
        total = (total + term)
    for i in range(8):
        total = (total * total)
    return total

def predict(x):
    W = [[[2.0]], [[3.0]]]
    B = [[-1.0], [0.5]]
    a = [x[0]]
    for l in range(len(W)):
        h = []
        for j in range(len(B[l])):
            z = B[l][j]
            for i in range(len(a)):
                z = (z + (a[i] * W[l][i][j]))
            if l < (len(W) - 1):
                if z >= 0.0:
                    e = exponential(-z)
                    h.append((1.0 / (1.0 + e)))
                else:
                    e = exponential(z)
                    h.append((e / (1.0 + e)))
            else:
                h.append(z)
        a = h
    return a[0]
```

<a id="example-mlpt"></a>

## 13 MLPRegressor with tanh activation

The hidden unit is tanh(2*x[0] - 1), followed by 3*h + 0.5. The proposed program explicitly implements the current saturation thresholds and uses the same fully included arithmetic exponential approximation on [-40, 40]. This illustrates how to avoid a primitive exponential; it is not a proof of numerical equivalence for every input or a finalized activation policy.

### Current output

```text
def predict(x):
 E=2.718281828459045
 W=[[[2.00e+00]],[[3.00e+00]]]
 B=[[-1.00e+00],[5.00e-01]]
 a=[x[0]]
 for l in range(len(W)):
  h=[]
  for j in range(len(B[l])):
   z=B[l][j]
   for i in range(len(a)):
    z+=a[i]*W[l][i][j]
   if l<len(W)-1:
    if z>20:
     h.append(1)
    elif z<-20:
     h.append(-1)
    else:
     e=E**(2*z)
     h.append((e-1)/(e+1))
   else:
    h.append(z)
  a=h
 return a[0]
```

### Proposed output

```python
def exponential(z):
    t = (z / 256.0)
    total = 1.0
    term = 1.0
    for i in range(20):
        term = ((term * t) / (i + 1))
        total = (total + term)
    for i in range(8):
        total = (total * total)
    return total

def predict(x):
    W = [[[2.0]], [[3.0]]]
    B = [[-1.0], [0.5]]
    a = [x[0]]
    for l in range(len(W)):
        h = []
        for j in range(len(B[l])):
            z = B[l][j]
            for i in range(len(a)):
                z = (z + (a[i] * W[l][i][j]))
            if l < (len(W) - 1):
                if z > 20.0:
                    h.append(1.0)
                else:
                    if z < -20.0:
                        h.append(-1.0)
                    else:
                        e = exponential((2.0 * z))
                        h.append(((e - 1.0) / (e + 1.0)))
            else:
                h.append(z)
        a = h
    return a[0]
```

<a id="example-ar"></a>

## 14 Autoregressive linear forecast

Here x[0] is the most recent observation, y_lag_1. The fitted one-step rule is 1 + 2*x[0], so the executable description matches an ordinary linear regressor with those coefficients. Lag names and candidate names are reporting metadata.

### Current output

```text
SCHEMA canonical_nescience_time_series_model_v1
MODEL autoregressive_linear
TASK forecasting
NAME autoregressive
INPUTS y_lag_1
PARAMETERS
    n_features = 1
    learned_coefficients = true
RULE
    y_hat = 1.0
    y_hat += 2.0 * y_lag_1
    return y_hat
```

### Proposed output

```python
def predict(x):
    y = (1.0 + (2.0 * x[0]))
    return y
```

<a id="example-ma"></a>

## 15 Moving-average forecast

Here x[0] and x[1] are the two most recent observations. This is a rolling-average forecast, not the moving-average error component of an ARIMA model.

### Current output

```text
SCHEMA canonical_nescience_time_series_model_v1
MODEL moving_average
TASK forecasting
NAME moving_average_2
INPUTS y_lag_1, y_lag_2
PARAMETERS
    n_features = 2
    learned_coefficients = false
RULE
    y_hat = 0.0
    y_hat += 0.5 * y_lag_1
    y_hat += 0.5 * y_lag_2
    return y_hat
```

### Proposed output

```python
def predict(x):
    y = ((0.5 * x[0]) + (0.5 * x[1]))
    return y
```

<a id="example-es"></a>

## 16 Finite-window exponential smoothing

For a two-observation window and alpha 0.5, the library's normalized weights are 2/3 and 1/3. The proposed literals preserve the fitted binary floating-point values rather than rounding them to six decimal places. This is the finite-window weighted predictor used by the searcher, not an infinite-history recursive smoother.

### Current output

```text
SCHEMA canonical_nescience_time_series_model_v1
MODEL exponential_smoothing
TASK forecasting
NAME exponential_smoothing_w2_a0.5
INPUTS y_lag_1, y_lag_2
PARAMETERS
    n_features = 2
    learned_coefficients = false
RULE
    y_hat = 0.0
    y_hat += 0.666667 * y_lag_1
    y_hat += 0.333333 * y_lag_2
    return y_hat
```

### Proposed output

```python
def predict(x):
    y = ((0.6666666666666666 * x[0]) + (0.3333333333333333 * x[1]))
    return y
```

<a id="example-arima"></a>

## 17 ARIMA with order (1, 0, 0)

This specific ARIMA(1, 0, 0) has coefficient 0.75, no intercept, and innovation variance 1. It is an AR(1) process with known initial prediction 0. For point predictions, the variance does not affect this particular recurrence and is therefore omitted from the executable program. This is not a serializer for arbitrary (p, d, q): differencing and moving-average terms require their own state and recurrences. initial_state() starts historical one-step filtering; step() returns the prediction BEFORE assimilating the supplied observation and then the updated state. forecast() advances that state without a new observation.

### Current output

```text
SCHEMA canonical_nescience_time_series_model_v1
MODEL arima
TASK forecasting
NAME arima_1_0_0
PARAMETERS
    order = (1, 0, 0)
    trend = n
    k_states = 1
    llf = -9.422818
    aic = 22.845635
    bic = 22.064511
COEFFICIENTS
    ar.L1 = 0.75
    sigma2 = 1.0
RULE
    estimate state-space representation with maximum likelihood
    return one_step_ahead_prediction
```

### Proposed output

```python
def initial_state():
    return [0.0]

def step(observation, state):
    prediction = state[0]
    next_state = [(0.75 * observation)]
    return [prediction, next_state]

def forecast(state):
    prediction = state[0]
    next_state = [(0.75 * prediction)]
    return [prediction, next_state]
```

<a id="example-level"></a>

## 18 State-space local level

The observation noise variance is 0.5 and the level innovation variance is 0.25. State is [predicted_level, predicted_variance], initialized to [0, 1]. step() performs the scalar Kalman update and then predicts the next state. forecast() advances without an observed value. These examples use known initialization, not the searcher's default diffuse initialization; a production serializer must encode the initialization actually used by its fitted model.

### Current output

```text
SCHEMA canonical_nescience_time_series_model_v1
MODEL state_space
TASK forecasting
NAME level
PARAMETERS
    specification = local_level
    k_states = 1
    states = level
    llf = -8.057618
    aic = 20.115236
    bic = 19.334112
COEFFICIENTS
    sigma2.irregular = 0.5
    sigma2.level = 0.25
RULE
    filter latent state with Kalman recursion
    return one_step_ahead_prediction
```

### Proposed output

```python
def initial_state():
    return [0.0, 1.0]

def step(observation, state):
    a = state[0]
    p = state[1]
    gain = (p / (p + 0.5))
    next_a = (a + (gain * (observation - a)))
    next_p = (((1.0 - gain) * p) + 0.25)
    return [a, [next_a, next_p]]

def forecast(state):
    return [state[0], [state[0], (state[1] + 0.25)]]
```

<a id="example-trend"></a>

## 19 State-space local linear trend

The observation, level, and trend innovation variances are 0.5, 0.25, and 0.125. State is [predicted_level, predicted_trend, P00, P01, P11]; the symmetric covariance initially equals the identity. The scalar observation update is followed by the level/trend transition. The known initialization is explicit. The complete program contains both filtering and forecasting; it does not call a matrix library.

### Current output

```text
SCHEMA canonical_nescience_time_series_model_v1
MODEL state_space
TASK forecasting
NAME trend
PARAMETERS
    specification = local_linear_trend
    k_states = 2
    states = level, trend
    llf = -5.821578
    aic = 17.643156
    bic = 15.802039
COEFFICIENTS
    sigma2.irregular = 0.5
    sigma2.level = 0.25
    sigma2.trend = 0.125
RULE
    filter latent state with Kalman recursion
    return one_step_ahead_prediction
```

### Proposed output

```python
def initial_state():
    return [0.0, 0.0, 1.0, 0.0, 1.0]

def step(observation, state):
    a = state[0]
    b = state[1]
    p00 = state[2]
    p01 = state[3]
    p11 = state[4]
    error = (observation - a)
    variance = (p00 + 0.5)
    k0 = (p00 / variance)
    k1 = (p01 / variance)
    filtered_a = (a + (k0 * error))
    filtered_b = (b + (k1 * error))
    f00 = ((1.0 - k0) * p00)
    f01 = ((1.0 - k0) * p01)
    f11 = (p11 - (k1 * p01))
    next_a = (filtered_a + filtered_b)
    next_p00 = (((f00 + (2.0 * f01)) + f11) + 0.25)
    next_p01 = (f01 + f11)
    next_p11 = (f11 + 0.125)
    return [a, [next_a, filtered_b, next_p00, next_p01, next_p11]]

def forecast(state):
    next_a = (state[0] + state[1])
    next_p00 = (((state[2] + (2.0 * state[3])) + state[4]) + 0.25)
    next_p01 = (state[3] + state[4])
    next_p11 = (state[4] + 0.125)
    return [state[0], [next_a, state[1], next_p00, next_p01, next_p11]]
```

## Verification

All 19 proposed programs were parsed, checked against an AST allowlist for the proposed constructs, compiled, and executed with only len and range available as built-in functions.

Stateless examples were checked on seven scalar inputs against their stated mathematical predictors. The multi-output case was checked component by component. The smoothing example was checked against the library's fitted weight values.

The ARIMA and both structural state-space examples were checked against statsmodels on the observation sequence [1, 2, 1.5, 3, 2.5, 4], followed by three forecast steps, using identical fitted parameters and known initialization. Absolute and relative tolerances were 1e-12.

The illustrative exponential helper was checked at 801 evenly spaced arguments between -40 and 40. The largest observed relative error was approximately 1.48e-13. This is a numerical test, not an error bound for all inputs. The logistic and tanh examples require a production numerical-domain and accuracy policy before this approach can become an accepted serializer.

## Design boundaries

These are minimal representative instances, not a complete implementation of each model family's serializer. Multiclass scoring, larger trees, wider networks, other ARIMA orders, and the exact initialization used by a fitted state-space model must also be generated and tested.

One grammar does not by itself fix a unique encoding. A production emitter should also standardize identifiers, whitespace, numeric precision, and arithmetic templates. The identical proposed affine descriptions for linear regression, linear SVR, and the one-lag autoregressive example illustrate that the measured program describes the predictor rather than the estimator's training algorithm.

No library implementation was changed to produce this comparison.
