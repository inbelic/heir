#ifndef LIB_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_PASSES_H_
#define LIB_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_PASSES_H_

#include "lib/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"
#include "lib/Dialect/ModArith/Conversions/ModArithToArith/PopulateModArith.h"
#include "lib/Dialect/ModArith/Conversions/ModArithToArith/PopulateModArith.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"

namespace mlir {
namespace heir {
namespace mod_arith {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/ModArith/Conversions/ModArithToArith/Passes.h.inc"

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_PASSES_H_
