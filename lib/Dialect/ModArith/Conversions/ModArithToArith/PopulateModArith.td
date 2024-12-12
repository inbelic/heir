#ifndef LIB_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_POPULATEMODARITH_TD_
#define LIB_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_POPULATEMODARITH_TD_

include "lib/Dialect/ModArith/IR/ModArithOps.td"
include "lib/Utils/DRR/Utils.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/IR/PatternBase.td"

def GetIntType : NativeCodeCall<"modulusTypeFromType($_builder.getContext(), $0.getType(), false)">;
def GetCModAttr : NativeCodeCall<"modulusAttrFromType($_builder.getContext(), $0.getType(), false)">;

def GetMulIntType : NativeCodeCall<"modulusTypeFromType($_builder.getContext(), $0.getType(), true)">;
def GetMulModType : NativeCodeCall<"modulusMulConvertType($_builder.getContext(), $0.getType())">;
def CreateConstant : NativeCodeCall<"$_builder.create<arith::ConstantOp>($_loc, $0)">;

def PopulateAdd : Pattern<
  (ModArith_AddOp:$modAdd $lhs, $rhs),
  [
    (ModArith_ExtractOp:$lhs_ext $lhs, (returnType (GetIntType $lhs))),
    (ModArith_ExtractOp:$rhs_ext $rhs, (returnType (GetIntType $rhs))),
    (Arith_AddIOp:$add $lhs_ext, $rhs_ext, DefOverflow),
    (ModArith_EncapsulateOp:$enc $add, (returnType $modAdd)),
    (ModArith_SubIfGEOp $enc)
  ]>;

def PopulateSub : Pattern<
  (ModArith_SubOp:$modSub $lhs, $rhs),
  [
    (ModArith_ExtractOp:$lhs_ext $lhs, (returnType (GetIntType $lhs))),
    (ModArith_ExtractOp:$rhs_ext $rhs, (returnType (GetIntType $rhs))),
    (Arith_SubIOp:$sub $lhs_ext, $rhs_ext, DefOverflow),
    (Arith_AddIOp:$shift $sub, (CreateConstant (GetCModAttr $lhs)), DefOverflow),
    (ModArith_EncapsulateOp:$enc $shift, (returnType $modSub)),
    (ModArith_SubIfGEOp $enc)
  ]>;

def PopulateMul : Pattern<
  (ModArith_MulOp:$modMul $lhs, $rhs),
  [
    (ModArith_ConvertOp:$lhs_conv $lhs, (returnType (GetMulModType $modMul))),
    (ModArith_ConvertOp:$rhs_conv $rhs, (returnType (GetMulModType $modMul))),
    (ModArith_ExtractOp:$lhs_ext $lhs_conv, (returnType (GetMulIntType $lhs))),
    (ModArith_ExtractOp:$rhs_ext $rhs_conv, (returnType (GetMulIntType $rhs))),
    (Arith_MulIOp:$mul $lhs_ext, $rhs_ext, DefOverflow),
    (ModArith_EncapsulateOp:$enc $mul, (returnType (GetMulModType $modMul))),
    (ModArith_BarrettReduceOp:$barrett $enc),
    (ModArith_SubIfGEOp:$res $enc),
    (ModArith_ConvertOp $res)
  ]>;

def PopulateMac : Pattern<
  (ModArith_MacOp:$modMac $lhs, $rhs, $acc),
  [
    (ModArith_MulOp:$modMul $lhs, $rhs),
    (ModArith_AddOp:$modAdd $modMul, $acc),
  ]>;

#endif  // LIB_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_POPULATEMODARITH_TD_
