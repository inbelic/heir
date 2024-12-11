// RUN: heir-opt -mod-arith-to-arith --split-input-file %s | FileCheck %s --enable-var-scope

!Zp = !mod_arith.int<65537 : i32>
!Zpv = tensor<4x!Zp>

// CHECK-LABEL: @test_lower_encapsulate
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_encapsulate(%lhs : i32) -> !Zp {
  // CHECK-NOT: mod_arith.encapsulate
  // CHECK: return %[[LHS]] : [[T]]
  %res = mod_arith.encapsulate %lhs: i32 -> !Zp
  return %res : !Zp
}

// CHECK-LABEL: @test_lower_encapsulate_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_encapsulate_vec(%lhs : tensor<4xi32>) -> !Zpv {
  // CHECK-NOT: mod_arith.encapsulate
  // CHECK: return %[[LHS]] : [[T]]
  %res = mod_arith.encapsulate %lhs: tensor<4xi32> -> !Zpv
  return %res : !Zpv
}

// CHECK-LABEL: @test_lower_extract
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_extract(%lhs : !Zp) -> i32 {
  // CHECK-NOT: mod_arith.extract
  // CHECK: return %[[LHS]] : [[T]]
  %res = mod_arith.extract %lhs: !Zp -> i32
  return %res : i32
}

// CHECK-LABEL: @test_lower_extract_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_extract_vec(%lhs : !Zpv) -> tensor<4xi32> {
  // CHECK-NOT: mod_arith.extract
  // CHECK: return %[[LHS]] : [[T]]
  %res = mod_arith.extract %lhs: !Zpv -> tensor<4xi32>
  return %res : tensor<4xi32>
}

// CHECK-LABEL: @test_lower_reduce
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_reduce(%lhs : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.reduce
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[T]]
  // CHECK: %[[REMS:.*]] = arith.remsi %[[LHS]], %[[CMOD]] : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[REMS]], %[[CMOD]] : [[T]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: return %[[REM]] : [[T]]
  %res = mod_arith.reduce %lhs: !Zp
  return %res : !Zp
}

// CHECK-LABEL: @test_lower_reduce_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_reduce_vec(%lhs : !Zpv) -> !Zpv {
  // CHECK-NOT: mod_arith.reduce
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[T]]
  // CHECK: %[[REMS:.*]] = arith.remsi %[[LHS]], %[[CMOD]] : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[REMS]], %[[CMOD]] : [[T]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: return %[[REM]] : [[T]]
  %res = mod_arith.reduce %lhs: !Zpv
  return %res : !Zpv
}

// CHECK-LABEL: @test_lower_add
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_add(%lhs : !Zp, %rhs : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.add
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: return %[[REM]] : [[T]]
  %res = mod_arith.add %lhs, %rhs : !Zp
  return %res : !Zp
}

// CHECK-LABEL: @test_lower_add_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_add_vec(%lhs : !Zpv, %rhs : !Zpv) -> !Zpv {
  // CHECK-NOT: mod_arith.add
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: return %[[REM]] : [[T]]
  %res = mod_arith.add %lhs, %rhs : !Zpv
  return %res : !Zpv
}

// CHECK-LABEL: @test_lower_sub
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_sub(%lhs : !Zp, %rhs : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.sub
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[T]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[SUB]], %[[CMOD]] : [[T]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: return %[[REM]] : [[T]]
  %res = mod_arith.sub %lhs, %rhs : !Zp
  return %res : !Zp
}

// CHECK-LABEL: @test_lower_sub_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_sub_vec(%lhs : !Zpv, %rhs : !Zpv) -> !Zpv {
  // CHECK-NOT: mod_arith.sub
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[T]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[SUB]], %[[CMOD]] : [[T]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: return %[[REM]] : [[T]]
  %res = mod_arith.sub %lhs, %rhs : !Zpv
  return %res : !Zpv
}

// CHECK-LABEL: @test_lower_mul
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_mul(%lhs : !Zp, %rhs : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.mul
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[TEXT:.*]]
  // CHECK: %[[EXT0:.*]] = arith.extui %[[LHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[EXT1:.*]] = arith.extui %[[RHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[MUL:.*]] = arith.muli %[[EXT0]], %[[EXT1]] : [[TEXT]]
  // CHECK: %[[REM:.*]] = arith.remui %[[MUL]], %[[CMOD]] : [[TEXT]]
  // CHECK: %[[TRUNC:.*]] = arith.trunci %[[REM]] : [[TEXT]] to [[T]]
  // CHECK: return %[[TRUNC]] : [[T]]
  %res = mod_arith.mul %lhs, %rhs : !Zp
  return %res : !Zp
}

// CHECK-LABEL: @test_lower_mul_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_mul_vec(%lhs : !Zpv, %rhs : !Zpv) -> !Zpv {
  // CHECK-NOT: mod_arith.mul
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[TEXT:.*]]
  // CHECK: %[[EXT0:.*]] = arith.extui %[[LHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[EXT1:.*]] = arith.extui %[[RHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[MUL:.*]] = arith.muli %[[EXT0]], %[[EXT1]] : [[TEXT]]
  // CHECK: %[[REM:.*]] = arith.remui %[[MUL]], %[[CMOD]] : [[TEXT]]
  // CHECK: %[[TRUNC:.*]] = arith.trunci %[[REM]] : [[TEXT]] to [[T]]
  // CHECK: return %[[TRUNC]] : [[T]]
  %res = mod_arith.mul %lhs, %rhs : !Zpv
  return %res : !Zpv
}

// CHECK-LABEL: @test_lower_mac
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]], %[[ACC:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_mac(%lhs : !Zp, %rhs : !Zp, %acc : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.mac
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[TEXT:.*]]
  // CHECK: %[[EXT0:.*]] = arith.extui %[[LHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[EXT1:.*]] = arith.extui %[[RHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[EXT2:.*]] = arith.extui %[[ACC]] : [[T]] to [[TEXT]]
  // CHECK: %[[MUL:.*]] = arith.muli %[[EXT0]], %[[EXT1]] : [[TEXT]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[MUL]], %[[EXT2]] : [[TEXT]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[TEXT]]
  // CHECK: %[[TRUNC:.*]] = arith.trunci %[[REM]] : [[TEXT]] to [[T]]
  // CHECK: return %[[TRUNC]] : [[T]]
  %res = mod_arith.mac %lhs, %rhs, %acc : !Zp
  return %res : !Zp
}

// CHECK-LABEL: @test_lower_mac_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]], %[[ACC:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_mac_vec(%lhs : !Zpv, %rhs : !Zpv, %acc : !Zpv) -> !Zpv {
  // CHECK-NOT: mod_arith.mac
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[TEXT:.*]]
  // CHECK: %[[EXT0:.*]] = arith.extui %[[LHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[EXT1:.*]] = arith.extui %[[RHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[EXT2:.*]] = arith.extui %[[ACC]] : [[T]] to [[TEXT]]
  // CHECK: %[[MUL:.*]] = arith.muli %[[EXT0]], %[[EXT1]] : [[TEXT]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[MUL]], %[[EXT2]] : [[TEXT]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[TEXT]]
  // CHECK: %[[TRUNC:.*]] = arith.trunci %[[REM]] : [[TEXT]] to [[T]]
  // CHECK: return %[[TRUNC]] : [[T]]
  %res = mod_arith.mac %lhs, %rhs, %acc : !Zpv
  return %res : !Zpv
}

// CHECK-LABEL: @test_lower_subifge
// CHECK-SAME: (%[[X:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_subifge(%x : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.subifge
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[T]]

  // CHECK: %[[SUB:.*]] = arith.subi %[[X]], %[[CMOD]] : [[T]]
  // CHECK: %[[CMP:.*]] = arith.cmpi sge, %[[X]], %[[CMOD]] : [[T]]
  // CHECK: %[[RES:.*]] = arith.select %[[CMP]], %[[SUB]], %[[X]] : [[T]]
  %res = mod_arith.subifge %x : !Zp
  return %res : !Zp
}

// CHECK-LABEL: @test_lower_subifge_vec
// CHECK-SAME: (%[[X:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_subifge_vec(%x : !Zpv) -> !Zpv {
  // CHECK-NOT: mod_arith.subifge
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[T]]

  // CHECK: %[[SUB:.*]] = arith.subi %[[X]], %[[CMOD]] : [[T]]
  // CHECK: %[[CMP:.*]] = arith.cmpi sge, %[[X]], %[[CMOD]] : [[T]]
  // CHECK: %[[RES:.*]] = arith.select %[[CMP]], %[[SUB]], %[[X]] : tensor<4xi1>, [[T]]
  %res = mod_arith.subifge %x : !Zpv
  return %res : !Zpv
}

// CHECK-LABEL: @test_lower_barrett_reduce
// CHECK-SAME: (%[[X:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_barrett_reduce(%arg : !Zp) -> !Zp {

  // CHECK: %[[RATIO:.*]] = arith.constant 262140 : [[INTER_TYPE:.*]]
  // CHECK: %[[BITWIDTH:.*]] = arith.constant 34 : [[INTER_TYPE]]
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[INTER_TYPE]]

  // CHECK: %[[EXT:.*]] = arith.extui %[[X]] : [[T]] to [[INTER_TYPE]]
  // CHECK: %[[MULRATIO:.*]] = arith.muli %[[EXT]], %[[RATIO]] : [[INTER_TYPE]]
  // CHECK: %[[SHIFTED:.*]] = arith.shrui %[[MULRATIO]], %[[BITWIDTH]] : [[INTER_TYPE]]
  // CHECK: %[[MULCMOD:.*]] = arith.muli %[[SHIFTED]], %[[CMOD]] : [[INTER_TYPE]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[EXT]], %[[MULCMOD]] : [[INTER_TYPE]]
  // CHECK: %[[RES:.*]] = arith.trunci %[[SUB]] : [[INTER_TYPE]] to [[T]]
  %res = mod_arith.barrett_reduce %arg : !Zp

  // CHECK: return %[[RES]] : [[T]]
  return %res : !Zp
}

// CHECK-LABEL: @test_lower_barrett_reduce_vec
// CHECK-SAME: (%[[X:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_barrett_reduce_vec(%arg : !Zpv) -> !Zpv {

  // CHECK: %[[RATIO:.*]] = arith.constant dense<262140> : [[INTER_TYPE:.*]]
  // CHECK: %[[BITWIDTH:.*]] = arith.constant dense<34> : [[INTER_TYPE]]
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[INTER_TYPE]]

  // CHECK: %[[EXT:.*]] = arith.extui %[[X]] : [[T]] to [[INTER_TYPE]]
  // CHECK: %[[MULRATIO:.*]] = arith.muli %[[EXT]], %[[RATIO]] : [[INTER_TYPE]]
  // CHECK: %[[SHIFTED:.*]] = arith.shrui %[[MULRATIO]], %[[BITWIDTH]] : [[INTER_TYPE]]
  // CHECK: %[[MULCMOD:.*]] = arith.muli %[[SHIFTED]], %[[CMOD]] : [[INTER_TYPE]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[EXT]], %[[MULCMOD]] : [[INTER_TYPE]]
  // CHECK: %[[RES:.*]] = arith.trunci %[[SUB]] : [[INTER_TYPE]] to [[T]]
  %res = mod_arith.barrett_reduce %arg : !Zpv

  // CHECK: return %[[RES]] : [[T]]
  return %res : !Zpv
}
