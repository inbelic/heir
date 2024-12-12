load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Conversions",
    hdrs = [
        "Passes.h",
    ],
    deps = [
        ":ModArithToArith",
        ":PopulateModArith",
        ":pass_inc_gen",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
    ],
)

cc_library(
    name = "ModArithToArith",
    srcs = ["ModArithToArith.cpp"],
    hdrs = [
        "ModArithToArith.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@heir//lib/Utils/ConversionUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "PopulateModArith",
    srcs = ["PopulateModArith.cpp"],
    hdrs = [
        "PopulateModArith.h",
    ],
    deps = [
        ":pass_inc_gen",
        ":populate_mod_arith_inc_gen",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@heir//lib/Utils/ConversionUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=ModArithToArith",
            ],
            "Passes.h.inc",
        ),
        (
            ["-gen-rewriters"],
            "Passes.cpp.inc",
        ),
        (
            ["-gen-pass-doc"],
            "ModArithToArith.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@heir//lib/Dialect/ModArith/IR:ops_inc_gen",
        "@heir//lib/Dialect/ModArith/IR:td_files",
        "@heir//lib/Utils/DRR",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "populate_mod_arith_inc_gen",
    includes = ["../../../../.."],
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "PopulateModArith.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PopulateModArith.td",
    deps = [
        "@heir//lib/Dialect/ModArith/IR:td_files",
        "@heir//lib/Utils/DRR",
        "@llvm-project//mlir:ArithOpsTdFiles",
    ],
)
