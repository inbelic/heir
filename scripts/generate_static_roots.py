"""Generate DenseMaps containing primitive roots of unity needed for NTTs."""

import argparse
import pathlib

import sympy


HEADER = """// WARNING: this file is autogenerated. Do not edit manually, instead see
// scripts/generate_static_roots.py

#ifndef LIB_DIALECT_POLYNOMIAL_IR_STATICROOTS_H_
#define LIB_DIALECT_POLYNOMIAL_IR_STATICROOTS_H_

#include <optional>

#include "llvm/include/llvm/ADT/APInt.h" // from @llvm-project
#include "llvm/include/llvm/ADT/DenseMap.h" // from @llvm-project

namespace roots {
"""

OPENING_TEMPLATE = """

llvm::DenseMap<std::pair<unsigned long long, unsigned long long>,
               unsigned long long>
    STATIC_{size}BIT_ROOTS({
"""

ENTRY_TEMPLATE = """        {{{{{cmod}, {degree}}}, {root}}},
"""

CLOSING = """    });
"""

FOOTER = """
// Attempts to find a 64-bit primitive 2n-th root of unity from the pre-computed
// values, where n is the given degree. find64BitRoot should be used if the
// required bits to represent cMod is greater than 32 and less than or equal to
// 64.
std::optional<llvm::APInt> find64BitRoot(const llvm::APInt& cMod,
                                         unsigned degree, unsigned bitWidth) {
  std::optional<llvm::APInt> root = std::nullopt;
  // We currently only precompute 64 bit and 32 bit values so we can skip
  // greater than that to ensure getZExtValue does not throw an error
  if (cMod.getBitWidth() <= 64) {
    auto rootIt = STATIC_64BIT_ROOTS.find({cMod.getZExtValue(), degree});
    if (rootIt != STATIC_64BIT_ROOTS.end())
      root = llvm::APInt(64, rootIt->second).trunc(bitWidth);
  }
  return root;
}

// Attempts to find a 32-bit primitive 2n-th root of unity from the pre-computed
// values, where n is the given degree. find32BitRoot should be used if the
// required bits to represent cMod is and less than or equal to 32.
std::optional<llvm::APInt> find32BitRoot(const llvm::APInt& cMod,
                                         unsigned degree, unsigned bitWidth) {
  std::optional<llvm::APInt> root = std::nullopt;
  // We currently only precompute 64 bit and 32 bit values so we can skip
  // greater than that to ensure getZExtValue does not throw an error
  if (cMod.getBitWidth() <= 64) {
    auto rootIt = STATIC_32BIT_ROOTS.find({cMod.getZExtValue(), degree});
    if (rootIt != STATIC_32BIT_ROOTS.end())
      root = llvm::APInt(64, rootIt->second).trunc(bitWidth);
  }
  return root;
}

}  // namespace roots

#endif  // LIB_DIALECT_POLYNOMIAL_IR_STATICROOTS_H_
"""

parser = argparse.ArgumentParser(
    description='Generate a static DenseMap of the primitive roots'
)
parser.add_argument(
    '--cmods',
    metavar='q',
    type=int,
    nargs='*',
    help='A list of coefficient modulus values that should be computed.',
)
parser.add_argument(
    'degrees',
    metavar='d',
    type=int,
    nargs='*',
    help=(
        'A list of degrees that the root should be computed for each cmod.'
        ' Defaults to [256, 512, ..., 65536].'
    ),
)


def output_map(outfile, cmod_mapping, size, degrees):
  outfile.write(OPENING_TEMPLATE.format(size=size))
  for cmod, roots in cmod_mapping.items():
    for degree, root in zip(degrees, roots):
      if root:
        output = ENTRY_TEMPLATE.format(cmod=cmod, degree=degree, root=root)
        outfile.write(output)
  outfile.write(CLOSING)


def generate_roots(q, ns):
  return [primitive_2nth_root(n, q) for n in ns]


def primitive_nth_roots(n, q):
  return [
      x
      for x in sympy.ntheory.nthroot_mod(1, n, q, True)
      if sympy.ntheory.n_order(x, q) == n
  ]


def smallest_primitive_nth_root(n, q):
  res = primitive_nth_roots(n, q)
  return None if not res else res[0]


def primitive_2nth_root(n, q):
  roots = [
      sympy.ntheory.nthroot_mod(a, 2, q) for a in primitive_nth_roots(n, q)
  ]
  res = [x for x in roots if x]
  return None if not res else res[0]


DEFAULT_DEGREES = [
    # 2**8 -> 2**16
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
]

DEFAULT_CMODS = [
    # 32-bit primes
    65537,
    114689,
    147457,
    163841,
    557057,
    638977,
    737281,
    786433,
    1032193,
    1179649,
    1769473,
    1785857,
    2277377,
    2424833,
    2572289,
    2654209,
    2752513,
    2768897,
    8380417,
    2147565569,
    2148155393,
    2148384769,
    3221225473,
    3221241857,
    3758161921,
    # 64-bit primes
    4295049217,
    8590163969,
    17180295169,
    34359771137,
    68720066561,
    137439510529,
]


def main(args: argparse.Namespace) -> None:
  degrees = args.degrees if args.degrees else DEFAULT_DEGREES
  cmods = args.cmods if args.cmods else DEFAULT_CMODS

  cmods32 = [x for x in cmods if x < 2**32]
  cmod32_mapping = {}
  for q in cmods32:
    cmod32_mapping[q] = [primitive_2nth_root(d, q) for d in degrees]

  cmods64 = [x for x in cmods if 2**32 <= x <= 2**64]
  cmod64_mapping = {}
  for q in cmods64:
    cmod64_mapping[q] = [primitive_2nth_root(d, q) for d in degrees]

  heir_root = pathlib.Path(__file__).parent
  static_roots = '../include/Dialect/Polynomial/Transforms/StaticRoots.h'
  output_fn = heir_root / static_roots
  with open(output_fn.resolve(), 'w') as outfile:
    outfile.write(HEADER)
    output_map(outfile, cmod64_mapping, '64', degrees)
    output_map(outfile, cmod32_mapping, '32', degrees)
    outfile.write(FOOTER)


if __name__ == '__main__':
  main(parser.parse_args())
