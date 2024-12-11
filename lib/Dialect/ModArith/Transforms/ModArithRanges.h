#ifndef LIB_DIALECT_MODARITH_TRANSFORMS_MODARITHRANGES_H_
#define LIB_DIALECT_MODARITH_TRANSFORMS_MODARITHRANGES_H_

#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "mlir/include/mlir/Pass/Pass.h"               // from @llvm-project

namespace mlir {
namespace heir {
namespace mod_arith {

#define GEN_PASS_DECL_MODARITHRANGES
#include "lib/Dialect/ModArith/Transforms/Passes.h.inc"

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_MODARITH_TRANSFORMS_MODARITHRANGES_H_
