---
title: "Production OOP Patterns in ML: Interview Reference"
date: 2026-07-19T00:00:00+05:30
draft: true
tags: ["oop", "python", "software-design", "interview-prep"]
summary: "A tour of OOP design patterns — encapsulation, inheritance, polymorphism, composition — as applied to building production ML systems."
---

# Production OOP Patterns in ML: Interview Reference

---

## **Table of Contents**

1. [Classes and Objects](#classes-and-objects)
2. [Encapsulation: Hide Implementation, Expose Interface](#encapsulation-hide-implementation-expose-interface)
3. [Inheritance: Reuse Code Across Models](#inheritance-reuse-code-across-models)
4. [Polymorphism: Same Interface, Different Behavior](#polymorphism-same-interface-different-behavior)
5. [Abstraction with ABC: Enforce the Contract](#abstraction-with-abc-enforce-the-contract)
6. [Duck Typing vs. ABC](#duck-typing-vs-abc)
7. [Method Overriding](#method-overriding)
8. [Composition: Building Complex Objects](#composition-building-complex-objects)
9. [Attributes: Data in Objects](#attributes-data-in-objects)
10. [Namespace and Scope](#namespace-and-scope)
11. [Quick Reference](#quick-reference)
12. [Interview Talking Points](#interview-talking-points)
13. [Decision Tree: When to Use Each Concept](#decision-tree-when-to-use-each-concept)
14. [Key Takeaway](#key-takeaway)

---

## **Classes and Objects**

### What is a Class?
A class is a **blueprint for creating objects**. It defines:
- **Attributes** (data the object holds)
- **Methods** (functions the object can perform)

### What is an Object?
An object is a **concrete instance** of a class. Multiple objects can exist from the same class, each with different data.

### Production Example
```python
class Model:
    def __init__(self, model_id: str):
        self.model_id = model_id      # attribute
        self.is_fitted = False         # attribute
    
    def fit(self, X, y):              # method
        pass
    
    def predict(self, X):             # method
        pass

# Objects: three different prediction systems
fraud_detector = Model(model_id="fraud_xgb")
ctr_predictor = Model(model_id="ctr_nn")
budget_forecaster = Model(model_id="forecast_arima")

# Each trains separately
fraud_detector.fit(X_fraud, y_fraud)
```

**Why it matters:** Classes enforce structure. Every model follows the same `.fit()` + `.predict()` interface. KServe expects this contract.

---

## **Encapsulation: Hide Implementation, Expose Interface**

### What is Encapsulation?
Bundle data (attributes) and behavior (methods) in a single unit, controlling what users can access.

### The Problem
Without encapsulation, callers repeat internal logic:

```python
# Fragile: every caller does preprocessing
model = xgb.XGBClassifier()
model.load_model("model.pkl")
features = raw_input[["age", "income"]]
features = scaler.transform(features)
prediction = model.predict_proba(features)
```

**Risk:** One mistake propagates everywhere.

### The Solution
Hide preprocessing inside the class:

```python
class PredictionServer:
    def __init__(self, model_path: str):
        self._model = xgb.XGBClassifier()         # private
        self._model.load_model(model_path)
        self._scaler = joblib.load("scaler.pkl") # private
    
    def predict(self, raw_input: dict) -> dict:  # public
        """User-facing interface."""
        features = self._preprocess(raw_input)
        score = self._model.predict_proba(features)[0, 1]
        return {"fraud_prob": score}
    
    def _preprocess(self, raw_input):             # private
        """Implementation detail."""
        features = pd.DataFrame([raw_input])
        return self._scaler.transform(features)

# Caller only uses public method
server = PredictionServer("models/fraud.pkl")
result = server.predict({"age": 35, "income": 100000})
# Result: {"fraud_prob": 0.12}
```

**Why it matters:** If we change preprocessing tomorrow, we update once inside `_preprocess()`, not in 50 services.

**Private vs. Public convention:**
- `public_method()` — designed for external use
- `_private_method()` — internal implementation, users shouldn't call this

---

## **Inheritance: Reuse Code Across Models**

### What is Inheritance?
A child class inherits attributes and methods from a parent class, reducing duplication.

### The Pattern
```python
class BaseEstimator:
    def __init__(self, model_id: str):
        self.model_id = model_id
    
    def _validate_input(self, X):
        """Shared validation logic."""
        if X.shape[1] != len(self.feature_names):
            raise ValueError("Feature mismatch")


class XGBoostModel(BaseEstimator):
    def fit(self, X, y):
        self._model = xgb.XGBClassifier()
        self._model.fit(X, y)
    
    def predict(self, X):
        self._validate_input(X)  # inherited method
        return self._model.predict_proba(X)


class CatBoostModel(BaseEstimator):
    def fit(self, X, y):
        self._model = catboost.CatBoostClassifier()
        self._model.fit(X, y)
    
    def predict(self, X):
        self._validate_input(X)  # inherited method
        return self._model.predict_proba(X)
```

**Why it matters:** Write `_validate_input()` once. Both XGBoost and CatBoost inherit it. No duplication.

---

## **Polymorphism: Same Interface, Different Behavior**

### What is Polymorphism?
Different classes respond to the same method call in different ways.

### The Pattern
```python
class EnsembleModel:
    def __init__(self, models: list):
        self.models = models  # can contain any model type
    
    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)  # polymorphic call
            # XGBoost.fit() works differently than CatBoost.fit()
            # But both respond to the same interface
    
    def predict(self, X):
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))  # polymorphic call
        return np.mean(predictions, axis=0)


# Usage
xgb_model = XGBoostModel()
cat_model = CatBoostModel()
lgb_model = LightGBMModel()

ensemble = EnsembleModel([xgb_model, cat_model, lgb_model])
ensemble.fit(X_train, y_train)
scores = ensemble.predict(X_test)
```

**Why it matters:** Ensemble doesn't care what's inside. Drop in a new model type without touching ensemble code.

---

## **Abstraction with ABC: Enforce the Contract**

### Why ABC Exists
In Python, everything is dynamic. Without enforcement, you can create a class that looks complete but misses required methods.

```python
# Without ABC, nothing stops this
class BadModel:
    def fit(self, X, y):
        print("Training")
    # Missing predict() — bug discovered at runtime
```

**ABC fixes this by:** Preventing incomplete subclasses from being instantiated.

### How ABC Works
```python
from abc import ABC, abstractmethod

class ModelContract(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass


# This works — implements all abstract methods
class GoodModel(ModelContract):
    def fit(self, X, y):
        self._model = xgb.XGBClassifier()
        self._model.fit(X, y)
    
    def predict(self, X):
        return self._model.predict_proba(X)

model = GoodModel()  # OK


# This fails — missing predict()
class BadModel(ModelContract):
    def fit(self, X, y):
        pass
    # Missing predict()

model = BadModel()  # TypeError: Can't instantiate abstract class BadModel
```

**Error timing matters:** With ABC, you fail at class definition time, not at runtime.

### When to Use ABC
| Use Case | Decision |
|----------|----------|
| Small scripts, one-off analysis | Don't use ABC |
| Production system, multiple teams | Use ABC |
| Framework design, plug-and-play system | Use ABC |
| Large codebase, implicit contracts become chaos | Use ABC |

---

## **Duck Typing vs. ABC**

### What is Duck Typing?
"If it walks like a duck and quacks like a duck, it's a duck."

Python doesn't check type — it checks capability.

```python
class Dog:
    def speak(self):
        return "Woof"

class Human:
    def speak(self):
        return "Hello"

def talk_to(entity):
    print(entity.speak())  # Python never checks if entity is Dog or Human
    # It just calls .speak() and trusts it works

talk_to(Dog())      # Works
talk_to(Human())    # Works
talk_to(123)        # Fails at runtime: 'int' has no attribute 'speak'
```

### Duck Typing Characteristics
- **Implicit contract** — no formal requirement
- **Flexible** — add new types easily
- **Risky at scale** — bugs discovered at runtime

### ABC Characteristics
- **Explicit contract** — formal requirement
- **Enforced** — bugs caught at instantiation time
- **Rigid** — requires inheritance from ABC

### When to Use Which?

**Duck Typing works when:**
- Code is small and well-understood
- Team is small
- Requirements are stable

**ABC works when:**
- Multiple teams contribute
- Requirements change frequently
- You want strict architectural discipline
- Production reliability matters

### Hybrid Approach (Best for ML)
```python
from abc import ABC, abstractmethod

class MLModel(ABC):
    """Enforces core contract for all models."""
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass


class XGBoostBidder(MLModel):
    """Concrete implementation."""
    def fit(self, X, y):
        self._model = xgb.XGBClassifier()
        self._model.fit(X, y)
    
    def predict(self, X):
        return self._model.predict_proba(X)
    
    def explain(self):
        """Extra method: duck typing allows this."""
        return self._model.feature_importances_


# Usage respects ABC contract + ducks type on explain()
model = XGBoostBidder()
model.fit(X, y)
model.predict(X_test)
model.explain()  # optional — duck typing
```

**Why this works:** Core contract is enforced (ABC). Additional methods are flexible (duck typing).

---

## **Method Overriding**

### What is Method Overriding?
A subclass defines a method with the same name as a parent class method, replacing the parent's implementation.

```python
class BaseModel:
    def validate(self, X):
        """Base validation."""
        if X is None:
            raise ValueError("X cannot be None")


class RobustModel(BaseModel):
    def validate(self, X):
        """Override: more strict validation."""
        super().validate(X)  # call parent logic first
        if X.shape[0] < 100:
            raise ValueError("Minimum 100 samples required")
```

**Why it matters:** Subclasses specialize parent behavior without breaking the interface.

---

## **Composition: Building Complex Objects**

### What is Composition?
Include instances of other classes as attributes within a class.

```python
class PredictionPipeline:
    def __init__(self, preprocessor, model, postprocessor):
        self.preprocessor = preprocessor    # composed objects
        self.model = model
        self.postprocessor = postprocessor
    
    def predict(self, X):
        X = self.preprocessor.transform(X)
        pred = self.model.predict(X)
        pred = self.postprocessor.transform(pred)
        return pred
```

**Why it matters:** More flexible than inheritance. You can swap components at runtime.

---

## **Attributes: Data in Objects**

### What are Attributes?
Variables that belong to an object and describe its state.

```python
class Model:
    def __init__(self, model_id: str, version: str):
        self.model_id = model_id        # attribute
        self.version = version           # attribute
        self.trained_at = None           # attribute
        self.metrics = {}                # attribute
```

**Why it matters:** Track model metadata (version, timestamp, performance). Enables versioning and rollbacks.

---

## **Namespace and Scope**

### What is a Namespace?
A **mapping of names → objects**. Think: a dictionary of identifiers.

```python
# Each scope has its own namespace
def train_model():
    X = load_data()        # X is in local namespace
    y = load_labels()      # y is in local namespace
    return model

# After function returns, X and y are removed from that namespace
```

### What is Scope?
The **region of code where Python will look for a name**.

### LEGB Resolution Rule
Python searches for a name in this order:

1. **L**ocal (inside current function)
2. **E**nclosing (in outer function, for nested functions)
3. **G**lobal (module-level)
4. **B**uilt-in (Python's built-ins like `print`, `len`)

```python
x = 10  # Global scope

def outer():
    x = 20  # Enclosing scope (for inner function)
    
    def inner():
        x = 30  # Local scope
        print(x)  # Prints 30 (Local)
    
    inner()
    print(x)  # Prints 20 (Enclosing)

print(x)  # Prints 10 (Global)
```

**Why it matters:** Understanding scope prevents variable shadowing bugs in nested class methods.

---

## **Quick Reference**

| Term | Means | Example |
|------|-------|---------|
| Class | Template for objects | `class Model` |
| Object | Instance of a class | `fraud_detector = Model()` |
| Attribute | Data in an object | `model.version` |
| Method | Function in a class | `model.fit()` |
| Inheritance | Child reuses parent | `XGBoostModel(BaseModel)` |
| Encapsulation | Hide internals, expose interface | `def predict()` public, `_validate()` private |
| Polymorphism | Same interface, different behavior | All models respond to `.fit()` and `.predict()` |
| Abstraction (ABC) | Enforce required methods | `@abstractmethod` forces implementation |
| Duck Typing | Check capability, not type | If it has `.speak()`, call it |
| Composition | Include other objects as attributes | `Pipeline(model1, model2, model3)` |
| Method Overriding | Subclass replaces parent method | `RobustModel.validate()` overrides `BaseModel.validate()` |

---

## **Interview Talking Points**

### **"Tell me about inheritance in your production systems."**
*"Every model at Adform—fraud, RTB, forecasting—inherits from `BaseEstimator` that enforces `.fit()`, `.predict()`. This ensures consistency across domains. When we deploy to KServe, the container expects this interface. Inheritance reduces boilerplate ~60%."*

### **"How do you handle incomplete implementations?"**
*"We use abstract base classes (ABC) with `@abstractmethod`. If a model skips `.predict()`, the code fails at instantiation—we catch bugs at definition time, not runtime. In production, that discipline matters."*

### **"Duck typing vs. ABC—which do you prefer?"**
*"At small scale, duck typing is flexible. At Adform's scale, ABC enforces architectural discipline. Core contract is ABC (`.fit()`, `.predict()`). Optional methods are duck typing (`.explain()`, `.get_metadata()`). Hybrid approach."*

### **"How does encapsulation help in production?"**
*"Every model deployed to KServe is wrapped with a `.predict()` method that handles scaling, validation, fallback, logging. Teams call one method. If we upgrade preprocessing tomorrow, we change it once inside the wrapper, not in 50 places."*

### **"Tell me about polymorphism in your work."**
*"In our RTB simulator, different bidding strategies (fixed, learned, dynamic) all inherit from `BiddingStrategy`. The loop calls `.generate_bid()` without knowing which strategy runs. We A/B tested three new strategies by dropping them into the same harness."*

---

## **Decision Tree: When to Use Each Concept**

```
Building a model class?
├─ Will multiple teams use it?
│  └─ YES → Use ABC (enforce contract)
│  └─ NO → Use duck typing (stay flexible)
│
├─ Does it need internal preprocessing?
│  └─ YES → Use encapsulation (hide internals)
│
├─ Will you have multiple model types?
│  └─ YES → Use inheritance + polymorphism (same interface, different behavior)
│
└─ Will you combine multiple models?
   └─ YES → Use composition (include other objects as attributes)
```

---

## **Key Takeaway**

OOP solves production problems:
- **Classes** enforce structure and contracts
- **Inheritance** reduces boilerplate across models
- **Encapsulation** prevents breaking changes
- **Polymorphism** enables safe composition
- **ABC** enforces discipline; duck typing enables flexibility
- **Composition** is more flexible than inheritance for complex systems

Not academic—it makes codebases maintainable at scale.
