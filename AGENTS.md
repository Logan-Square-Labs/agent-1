## Code Design Guidelines

**Above all, write code that is extremely simple and consumable.**

### Writing New Code
- Design modules to be deep: simple interfaces, rich functionality behind them.
  Hide implementation details. If a class interface is nearly as complex as its
  implementation, reconsider the abstraction.
- Design interfaces for the common case. Rare use cases shouldn't complicate
  the API that everyday callers see.
- Pull complexity downward: it's better for a module's implementation to be
  complex than for its interface to be. Don't push complexity onto callers.
- Separate general-purpose code from special-purpose code. General-purpose
  utilities belong in their own modules, not interleaved with business logic.
- Different layers should use different abstractions. If two layers use the
  same vocabulary and data structures, they probably shouldn't be separate.
- Define errors out of existence where possible. Instead of throwing an
  exception for a common case, handle it internally.
- Write comments that explain *why* and what's not obvious from the code,
  not *what* the code does line by line.
- Design for reading, not writing. Code is read far more often than written.

### Red Flags to Watch For
- **Shallow module**: The interface is almost as complex as the implementation.
  Consolidate or rethink the abstraction.
- **Pass-through method**: A method that just forwards arguments to another
  method with a similar signature. Eliminate the indirection.
- **Information leakage**: The same design decision appears in multiple modules.
  Consolidate it behind one interface.
- **Temporal decomposition**: Code is structured around the order operations
  happen rather than around information hiding. Restructure around what
  information each module encapsulates.
- **Overexposure**: An API forces callers to deal with rarely-used features
  to access common ones. Provide sensible defaults.
- **Vague or hard-to-pick names**: If naming is difficult, the abstraction
  may be wrong. Rethink the boundaries.
- **Comment repeats code**: Delete comments that restate what the code
  obviously does. Replace with comments about intent or non-obvious behavior.