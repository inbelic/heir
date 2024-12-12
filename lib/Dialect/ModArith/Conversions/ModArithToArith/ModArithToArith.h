#ifndef LIB_DIALECT_MODARITH_TRANSFORMS_MODARITHTOARITH_H_
#define LIB_DIALECT_MODARITH_TRANSFORMS_MODARITHTOARITH_H_

#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"               // from @llvm-project

namespace mlir {
namespace heir {
namespace mod_arith {

#define GEN_PASS_DECL_MODARITHTOARITH
#include "lib/Dialect/ModArith/Conversions/ModArithToArith/Passes.h.inc"

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_MODARITH_TRANSFORMS_MODARITHTOARITH_H_
