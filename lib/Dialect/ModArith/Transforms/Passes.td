#ifndef LIB_DIALECT_MODARITH_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_MODARITH_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def ModArithRanges : Pass<"compute-mod-arith-ranges"> {
  let summary = "computes integer range analysis with mod_arith ops";
  let description = [{}];
  let dependentDialects = ["mlir::heir::mod_arith::ModArithDialect"];
}

#endif  // LIB_DIALECT_MODARITH_TRANSFORMS_PASSES_TD_
