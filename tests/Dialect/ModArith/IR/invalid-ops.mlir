// RUN: heir-opt --verify-diagnostics --split-input-file %s | FileCheck %s

!Zp = !mod_arith.int<255 : i8>

// CHECK-NOT: @test_bad_mod
func.func @test_bad_mod(%lhs : i8) -> !Zp {
  // expected-error@+1 {{underlying type's bitwidth must be 1 bit larger than the modulus bitwidth, but got 8 while modulus requires width 8.}}
  %m = mod_arith.encapsulate %lhs : i8 -> !Zp
  return %m : !Zp
}

// -----

!Zp = !mod_arith.int<255 : i32>

// CHECK-NOT: @test_bad_extract
func.func @test_bad_extract(%lhs : !Zp) -> i8 {
  // expected-error@+1 {{the result integer type should be of the same width as the mod arith type width, but got 8 while mod arith type width 32}}
  %m = mod_arith.extract %lhs : !Zp -> i8
  return %m : i8
}

// -----
