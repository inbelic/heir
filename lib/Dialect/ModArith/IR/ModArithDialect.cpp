#include "lib/Dialect/ModArith/IR/ModArithDialect.h"

#include <cassert>
#include <optional>

#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"       // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

// NOLINTBEGIN(misc-include-cleaner): Required to define ModArithDialect,
// ModArithTypes, ModArithOps, ModArithAttributes
#include "lib/Dialect/ModArith/IR/ModArithAttributes.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
// NOLINTEND(misc-include-cleaner)

// Generated definitions
#include "lib/Dialect/ModArith/IR/ModArithDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/ModArith/IR/ModArithAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/ModArith/IR/ModArithTypes.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/ModArith/IR/ModArithOps.cpp.inc"

namespace mlir {
namespace heir {
namespace mod_arith {

class ModArithOpAsmDialectInterface : public OpAsmDialectInterface {
 public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Type type, raw_ostream &os) const override {
    auto res = llvm::TypeSwitch<Type, AliasResult>(type)
                   .Case<ModArithType>([&](auto &modArithType) {
                     os << "Z";
                     os << modArithType.getModulus().getValue();
                     os << "_";
                     os << modArithType.getModulus().getType();
                     return AliasResult::FinalAlias;
                   })
                   .Default([&](Type) { return AliasResult::NoAlias; });
    return res;
  }
};

void ModArithDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/ModArith/IR/ModArithTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/ModArith/IR/ModArithAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/ModArith/IR/ModArithOps.cpp.inc"
      >();

  addInterface<ModArithOpAsmDialectInterface>();
}

/// Ensures that the underlying integer type is wide enough for the coefficient
template <typename OpType>
LogicalResult verifyModArithType(OpType op, ModArithType type) {
  APInt modulus = type.getModulus().getValue();
  unsigned bitWidth = modulus.getBitWidth();
  unsigned modWidth = modulus.getActiveBits();
  if (modWidth > bitWidth - 1)
    return op.emitOpError()
           << "underlying type's bitwidth must be 1 bit larger than "
           << "the modulus bitwidth, but got " << bitWidth
           << " while modulus requires width " << modWidth << ".";
  return success();
}

template <typename OpType>
LogicalResult verifySameWidth(OpType op, ModArithType modArithType,
                              IntegerType integerType) {
  unsigned bitWidth = modArithType.getModulus().getValue().getBitWidth();
  unsigned intWidth = integerType.getWidth();
  if (intWidth != bitWidth)
    return op.emitOpError()
           << "the result integer type should be of the same width as the "
           << "mod arith type width, but got " << intWidth
           << " while mod arith type width " << bitWidth << ".";
  return success();
}

LogicalResult EncapsulateOp::verify() {
  auto modArithType = getResultModArithType(*this);
  auto integerType = getOperandIntegerType(*this);
  auto result = verifySameWidth(*this, modArithType, integerType);
  if (result.failed()) return result;
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult ExtractOp::verify() {
  auto modArithType = getOperandModArithType(*this);
  auto integerType = getResultIntegerType(*this);
  auto result = verifySameWidth(*this, modArithType, integerType);
  if (result.failed()) return result;
  return verifyModArithType(*this, modArithType);
}

LogicalResult ReduceOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult AddOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult SubOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult MulOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult MacOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult BarrettReduceOp::verify() {
  auto inputType = getInput().getType();
  unsigned bitWidth;
  if (auto tensorType = dyn_cast<RankedTensorType>(inputType)) {
    bitWidth = tensorType.getElementTypeBitWidth();
  } else {
    auto integerType = dyn_cast<IntegerType>(inputType);
    assert(integerType &&
           "expected input to be a ranked tensor type or integer type");
    bitWidth = integerType.getWidth();
  }
  auto expectedBitWidth = (getModulus() - 1).getActiveBits();
  if (bitWidth < expectedBitWidth || 2 * expectedBitWidth < bitWidth) {
    return emitOpError()
           << "input bitwidth is required to be in the range [w, 2w], where w "
              "is the smallest bit-width that contains the range [0, modulus). "
              "Got "
           << bitWidth << " but w is " << expectedBitWidth << ".";
  }
  if (getModulus().slt(0))
    return emitOpError() << "provided modulus " << getModulus().getSExtValue()
                         << " is not a positive integer.";
  return success();
}

template <typename OpType>
IntegerValueRange initCanonicalRange(OpType op) {
  if (auto modType = dyn_cast<ModArithType>(op->getResult(0).getType())) {
    APInt q = modType.getModulus().getValue();
    return ConstantIntRanges::fromSigned(APInt(q.getBitWidth(), 0), q - 1);
  }
  // Doesn't currently support shaped types
  return IntegerValueRange();
}

template <typename OpType>
IntegerValueRange getCanonicalOpRanges(OpType op, ArrayRef<mlir::IntegerValueRange> inputRanges) {
  auto opRange = initCanonicalRange(op);
  if (opRange.isUninitialized()) return IntegerValueRange();
  auto q = opRange.getValue().smax();

  for (auto curRange : inputRanges) {
    // Ensure that the input ranges exist
    if (curRange.isUninitialized()) return IntegerValueRange();

    // Ensure that the input ranges are with the canocial range [0, q)
    APInt curMin = curRange.getValue().smin();
    APInt curMax = curRange.getValue().smax();
    if (curMin.getBitWidth() != q.getBitWidth()) return IntegerValueRange();
    if (curMin.slt(0) || curMax.sgt(q))
      return IntegerValueRange();
  }
  return opRange;
}

void EncapsulateOp::inferResultRangesFromOptional(
    ArrayRef<mlir::IntegerValueRange> inputRanges, SetIntLatticeFn setResultRange) {
  // NOTE: this will cause failure when the inputRanges is a ConstantValue. This is
  // because when invoking IntegerRangeValueLattice::onUpdate, we will try to update
  // the SSA value to the constant but since the node type is not an integer/index value
  // we will fail to create the IntegerAttr. This will work when the upstream is resolved
  // tracked here: ADD ISSUE HERE.
  setResultRange(getResult(), inputRanges[0]);
}

void ExtractOp::inferResultRangesFromOptional(
    ArrayRef<mlir::IntegerValueRange> inputRanges, SetIntLatticeFn setResultRange) {
  setResultRange(getResult(), inputRanges[0]);
}

void ReduceOp::inferResultRangesFromOptional(
    ArrayRef<mlir::IntegerValueRange> inputRanges, SetIntLatticeFn setResultRange) {
  setResultRange(getResult(), initCanonicalRange(*this));
}

void AddOp::inferResultRangesFromOptional(
    ArrayRef<mlir::IntegerValueRange> inputRanges, SetIntLatticeFn setResultRange) {
  setResultRange(getResult(), getCanonicalOpRanges(*this, inputRanges));
}

void SubOp::inferResultRangesFromOptional(
    ArrayRef<mlir::IntegerValueRange> inputRanges, SetIntLatticeFn setResultRange) {
  setResultRange(getResult(), getCanonicalOpRanges(*this, inputRanges));
}

void MulOp::inferResultRangesFromOptional(
    ArrayRef<mlir::IntegerValueRange> inputRanges, SetIntLatticeFn setResultRange) {
  setResultRange(getResult(), getCanonicalOpRanges(*this, inputRanges));
}

void MacOp::inferResultRangesFromOptional(
    ArrayRef<mlir::IntegerValueRange> inputRanges, SetIntLatticeFn setResultRange) {
  setResultRange(getResult(), getCanonicalOpRanges(*this, inputRanges));
}

void BarrettReduceOp::inferResultRangesFromOptional(
    ArrayRef<mlir::IntegerValueRange> inputRanges, SetIntLatticeFn setResultRange) {
  // TODO(#1084): update to using mod arith type
  if (inputRanges[0].isUninitialized()) return;
  auto q = getModulus();
  auto zero = APInt(q.getBitWidth(), 0);
  auto inputRange = inputRanges[0].getValue();
  if (inputRange.smin().slt(0) || inputRange.smax().sgt(q * q)) return;
  auto max = inputRange.smax().slt(q) ? q : 2 * q;
  auto outputRange = ConstantIntRanges::fromSigned(zero, max);
  setResultRange(getResult(), IntegerValueRange{outputRange});
}

void SubIfGEOp::inferResultRangesFromOptional(
    ArrayRef<mlir::IntegerValueRange> inputRanges, SetIntLatticeFn setResultRange) {
  // TODO(#1084): update to using mod arith type
  if (inputRanges[0].isUninitialized() || inputRanges[1].isUninitialized()) return;

  auto lhsRange = inputRanges[0].getValue();
  auto rhsRange = inputRanges[1].getValue();
  auto intersection = lhsRange.intersection(rhsRange);
  bool intersected = intersection.smin().sle(intersection.smax());

  auto lhsMin = lhsRange.smin();
  auto lhsMax = lhsRange.smax();
  auto rhsMin = rhsRange.smin();
  auto rhsMax = rhsRange.smax();

  // Default to the case that rhsMax < lhsMin and there will be a sub
  ConstantIntRanges range
     = ConstantIntRanges::fromSigned(lhsMin - rhsMax, rhsMax - lhsMin);
  if (lhsMax.slt(rhsMin)) // When there will not be a sub
    range = lhsRange;
  else if (intersected) // When there will be a potential sub for the intersecting values
    range = ConstantIntRanges::fromSigned(intersection.smin() - rhsMax, intersection.smin() - 1);
  
  setResultRange(getResult(), IntegerValueRange{range});
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  APInt parsedValue(64, 0);
  Type parsedType;

  if (failed(parser.parseInteger(parsedValue))) {
    parser.emitError(parser.getCurrentLocation(),
                     "found invalid integer value");
    return failure();
  }

  if (parser.parseColon() || parser.parseType(parsedType)) return failure();

  auto modArithType = dyn_cast<ModArithType>(parsedType);
  if (!modArithType) return failure();

  auto outputBitWidth =
      modArithType.getModulus().getType().getIntOrFloatBitWidth();
  if (parsedValue.getActiveBits() > outputBitWidth)
    return parser.emitError(parser.getCurrentLocation(),
                            "constant value is too large for the modulus");

  auto intValue = IntegerAttr::get(modArithType.getModulus().getType(),
                                   parsedValue.trunc(outputBitWidth));
  result.addAttribute(
      "value", ModArithAttr::get(parser.getContext(), modArithType, intValue));
  result.addTypes(modArithType);
  return success();
}

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  // getValue chain:
  // op's ModArithAttribute value
  //   -> ModArithAttribute's IntegerAttr value
  //   -> IntegerAttr's APInt value
  getValue().getValue().getValue().print(p.getStream(), true);
  p << " : ";
  p.printType(getOutput().getType());
}

LogicalResult ConstantOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> loc,
    ConstantOpAdaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  returnTypes.push_back(adaptor.getValue().getType());
  return success();
}

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir
