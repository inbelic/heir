#include "lib/Dialect/ModArith/Conversions/ModArithToArith/PopulateModArith.h"

#include <utility>

#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"       // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace mod_arith {

static TypedAttr modulusAttrFromType(MLIRContext *context, Type type,
                                     bool mul) {
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(type));
  APInt modulus = modArithType.getModulus().getValue();
  auto width = modulus.getBitWidth();
  width = mul ? 2 * width : width;

  auto intType = IntegerType::get(context, width);
  auto truncmod = modulus.zextOrTrunc(width);

  if (auto st = mlir::dyn_cast<ShapedType>(type)) {
    auto containerType = st.cloneWith(st.getShape(), intType);
    return DenseElementsAttr::get(containerType, truncmod);
  }
  return IntegerAttr::get(intType, truncmod);
}

static Type modulusTypeFromType(MLIRContext *context, Type type, bool mul) {
  return modulusAttrFromType(context, type, mul).getType();
}

static Type modulusMulConvertType(MLIRContext *context, Type type) {
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(type));
  APInt modulus = modArithType.getModulus().getValue();
  auto width = modulus.getBitWidth() * 2;

  auto intType = IntegerType::get(context, width);
  auto truncmod = modulus.zextOrTrunc(width);

  auto modType =
      ModArithType::get(context, IntegerAttr::get(intType, truncmod));

  if (auto st = mlir::dyn_cast<ShapedType>(type)) {
    return st.cloneWith(st.getShape(), modType);
  }
  return modType;
}

#define GEN_PASS_DEF_POPULATEMODARITH
#include "lib/Dialect/ModArith/Conversions/ModArithToArith/Passes.h.inc"

namespace rewrites {
// In an inner namespace to avoid conflicts with canonicalization patterns
#include "lib/Dialect/ModArith/Conversions/ModArithToArith/PopulateModArith.cpp.inc"
}  // namespace rewrites

struct PopulateModArith : impl::PopulateModArithBase<PopulateModArith> {
  using PopulateModArithBase::PopulateModArithBase;

  void runOnOperation() override;
};

void PopulateModArith::runOnOperation() {
  MLIRContext *context = &getContext();

  RewritePatternSet patterns(context);
  patterns.add<rewrites::PopulateAdd, rewrites::PopulateSub,
               rewrites::PopulateMul, rewrites::PopulateMac>(
      patterns.getContext());

  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir
