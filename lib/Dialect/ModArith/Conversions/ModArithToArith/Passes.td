#ifndef LIB_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_PASSES_TD_
#define LIB_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_PASSES_TD_

include "lib/Utils/DRR/Utils.td"
include "lib/Dialect/ModArith/IR/ModArithOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/IR/PatternBase.td"
include "mlir/Pass/PassBase.td"

def ModArithToArith : Pass<"mod-arith-to-arith", "ModuleOp"> {
  let summary = "Lower `mod_arith` to standard `arith`.";

  let description = [{
    This pass lowers the `mod_arith` dialect to their `arith` equivalents.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::mod_arith::ModArithDialect",
  ];
}

def PopulateModArith : Pass<"populate-mod-arith", "ModuleOp"> {
  let summary = "Populates the IR with `mod_arith` `subifge` and `barrett_reduce`.";

  let description = [{
    This pass goes partially lowers the following `mod_arith` ops to `arith` ops
    with `mod_arith.subifge` and `mod_arith.barrett_reduce` when possible.

    The converted ops are: `Add`, `Sub`, `Mul`, `Mac`.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::mod_arith::ModArithDialect",
  ];
}

#endif  // LIB_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_PASSES_TD_
