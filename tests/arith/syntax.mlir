// RUN: heir-opt %s | FileCheck %s

// CHECK-LABEL: @test_arith_syntax
func.func @test_arith_syntax() {
  %zero = arith.constant 1 : i8
  %cmod = arith.constant 17 : i8
  %c_vec = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi8>
  %cmod_vec = arith.constant dense<17> : tensor<4xi8>

  // CHECK: arith_ext.barrett_reduce
  // CHECK: arith_ext.barrett_reduce
  %barrett = arith_ext.barrett_reduce %zero { modulo = 17 } : i8
  %barrett_vec = arith_ext.barrett_reduce %c_vec { modulo = 17 } : tensor<4xi8>

  // CHECK: arith_ext.normalised
  // CHECK: arith_ext.normalised
  %normalised = arith_ext.normalised %zero { q = 17 } : i8
  %normalised_vec = arith_ext.normalised %c_vec { q = 17 } : tensor<4xi8>

  // CHECK: arith_ext.subifge
  // CHECK: arith_ext.subifge
  %subifge = arith_ext.subifge %zero, %cmod : i8
  %subifge_vec = arith_ext.subifge %c_vec, %cmod_vec : tensor<4xi8>

  return
}
