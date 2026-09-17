  .text
  .globl nn_dot_product_f64
  .type nn_dot_product_f64, @function

# double nn_dot_product_f64(const double* left, const double* right,
#                           std::size_t length)
# Arguments: rdi = left, rsi = right, rdx = length. Result: xmm0.
nn_dot_product_f64:
  xorpd %xmm0, %xmm0
  xorq  %rax, %rax
  testq %rdx, %rdx
  je    .Ldot_done

.Ldot_loop:
  movsd (%rdi, %rax, 8), %xmm1
  mulsd (%rsi, %rax, 8), %xmm1
  addsd %xmm1, %xmm0
  incq  %rax
  cmpq  %rdx, %rax
  jne   .Ldot_loop

.Ldot_done:
  ret

  .globl nn_relu_f64
  .type nn_relu_f64, @function

# double nn_relu_f64(double value)
# Argument/result: xmm0. NaN follows std::max(0.0, value) and returns +0.0.
nn_relu_f64:
  xorpd  %xmm1, %xmm1
  ucomisd %xmm1, %xmm0
  ja     .Lrelu_done
  xorpd  %xmm0, %xmm0
.Lrelu_done:
  ret

  .globl nn_relu_derivative_from_output_f64
  .type nn_relu_derivative_from_output_f64, @function

# double nn_relu_derivative_from_output_f64(double output)
# Argument/result: xmm0.
nn_relu_derivative_from_output_f64:
  xorpd  %xmm1, %xmm1
  ucomisd %xmm1, %xmm0
  ja     .Lrelu_derivative_positive
  xorpd  %xmm0, %xmm0
  ret
.Lrelu_derivative_positive:
  movsd .Lone(%rip), %xmm0
  ret

  .globl nn_exp_approx_f64
  .type nn_exp_approx_f64, @function

# double nn_exp_approx_f64(double value)
# Range-reduce value = k * ln(2) + r, then evaluate exp(r) with a ninth-degree
# Taylor polynomial. The intended input domain is [-16, 16].
nn_exp_approx_f64:
  movapd %xmm0, %xmm1
  mulsd  .Linv_ln2(%rip), %xmm1
  cvtsd2si %xmm1, %rax
  cvtsi2sd %rax, %xmm2
  mulsd  .Lln2(%rip), %xmm2
  subsd  %xmm2, %xmm0

  movsd .Lexp_c9(%rip), %xmm1
  mulsd %xmm0, %xmm1
  addsd .Lexp_c8(%rip), %xmm1
  mulsd %xmm0, %xmm1
  addsd .Lexp_c7(%rip), %xmm1
  mulsd %xmm0, %xmm1
  addsd .Lexp_c6(%rip), %xmm1
  mulsd %xmm0, %xmm1
  addsd .Lexp_c5(%rip), %xmm1
  mulsd %xmm0, %xmm1
  addsd .Lexp_c4(%rip), %xmm1
  mulsd %xmm0, %xmm1
  addsd .Lexp_c3(%rip), %xmm1
  mulsd %xmm0, %xmm1
  addsd .Lexp_c2(%rip), %xmm1
  mulsd %xmm0, %xmm1
  addsd .Lone(%rip), %xmm1
  mulsd %xmm0, %xmm1
  addsd .Lone(%rip), %xmm1

  addq $1023, %rax
  shlq $52, %rax
  movq %rax, %xmm2
  mulsd %xmm2, %xmm1
  movapd %xmm1, %xmm0
  ret

  .globl nn_sigmoid_f64
  .type nn_sigmoid_f64, @function

# double nn_sigmoid_f64(double value)
# Saturation at +/-14 keeps the maximum saturation error below 1e-6.
nn_sigmoid_f64:
  ucomisd %xmm0, %xmm0
  jp      .Lsigmoid_compute
  ucomisd .Lsigmoid_upper(%rip), %xmm0
  jae     .Lsigmoid_one
  ucomisd .Lsigmoid_lower(%rip), %xmm0
  jbe     .Lsigmoid_zero
.Lsigmoid_compute:
  xorpd  %xmm1, %xmm1
  subsd  %xmm0, %xmm1
  movapd %xmm1, %xmm0
  subq   $8, %rsp
  call   nn_exp_approx_f64
  addq   $8, %rsp
  addsd  .Lone(%rip), %xmm0
  movsd  .Lone(%rip), %xmm1
  divsd  %xmm0, %xmm1
  movapd %xmm1, %xmm0
  ret
.Lsigmoid_one:
  movsd .Lone(%rip), %xmm0
  ret
.Lsigmoid_zero:
  xorpd %xmm0, %xmm0
  ret

  .globl nn_sigmoid_derivative_from_output_f64
  .type nn_sigmoid_derivative_from_output_f64, @function

# double nn_sigmoid_derivative_from_output_f64(double output)
nn_sigmoid_derivative_from_output_f64:
  movsd .Lone(%rip), %xmm1
  subsd %xmm0, %xmm1
  mulsd %xmm1, %xmm0
  ret

  .globl nn_tanh_f64
  .type nn_tanh_f64, @function

# double nn_tanh_f64(double value)
# Saturation at +/-8 has an absolute error below 1e-6.
nn_tanh_f64:
  ucomisd %xmm0, %xmm0
  jp      .Ltanh_nan
  ucomisd .Ltanh_upper(%rip), %xmm0
  jae     .Ltanh_one
  ucomisd .Ltanh_lower(%rip), %xmm0
  jbe     .Ltanh_negative_one
  addsd   %xmm0, %xmm0
  subq    $8, %rsp
  call    nn_exp_approx_f64
  addq    $8, %rsp
  movapd  %xmm0, %xmm1
  subsd   .Lone(%rip), %xmm1
  addsd   .Lone(%rip), %xmm0
  divsd   %xmm0, %xmm1
  movapd  %xmm1, %xmm0
  ret
.Ltanh_one:
  movsd .Lone(%rip), %xmm0
  ret
.Ltanh_negative_one:
  movsd .Lnegative_one(%rip), %xmm0
  ret
.Ltanh_nan:
  subsd %xmm0, %xmm0
  ret

  .globl nn_tanh_derivative_from_output_f64
  .type nn_tanh_derivative_from_output_f64, @function

# double nn_tanh_derivative_from_output_f64(double output)
nn_tanh_derivative_from_output_f64:
  mulsd %xmm0, %xmm0
  movsd .Lone(%rip), %xmm1
  subsd %xmm0, %xmm1
  movapd %xmm1, %xmm0
  ret

  .globl nn_fast_layer_forward_f64
  .type nn_fast_layer_forward_f64, @function

# void nn_fast_layer_forward_f64(
#   const double* input, const double* weights, const double* biases,
#   double* output, size_t numInputs, size_t numOutputs, int activationKind)
# The seventh argument is passed on the stack under the System V AMD64 ABI.
nn_fast_layer_forward_f64:
  pushq %rbx
  pushq %r12
  pushq %r13
  pushq %r14
  pushq %r15
  subq  $32, %rsp
  movq  %rdi, %r12
  movq  %rsi, %r13
  movq  %rdx, %r14
  movq  %rcx, %r15
  movq  %r8, %rbx
  movq  %r9, 0(%rsp)
  movl  80(%rsp), %eax
  movl  %eax, 8(%rsp)
  movq  $0, 16(%rsp)

.Lfast_forward_loop:
  movq  16(%rsp), %rax
  cmpq  0(%rsp), %rax
  jae   .Lfast_forward_done
  imulq %rbx, %rax
  leaq  (%r13, %rax, 8), %rdi
  movq  %r12, %rsi
  movq  %rbx, %rdx
  call  nn_dot_product_f64
  movq  16(%rsp), %rax
  addsd (%r14, %rax, 8), %xmm0

  cmpl  $0, 8(%rsp)
  je    .Lfast_forward_sigmoid
  cmpl  $1, 8(%rsp)
  je    .Lfast_forward_tanh
  call  nn_relu_f64
  jmp   .Lfast_forward_store
.Lfast_forward_sigmoid:
  call  nn_sigmoid_f64
  jmp   .Lfast_forward_store
.Lfast_forward_tanh:
  call  nn_tanh_f64
.Lfast_forward_store:
  movq  16(%rsp), %rax
  movsd %xmm0, (%r15, %rax, 8)
  incq  %rax
  movq  %rax, 16(%rsp)
  jmp   .Lfast_forward_loop

.Lfast_forward_done:
  addq  $32, %rsp
  popq  %r15
  popq  %r14
  popq  %r13
  popq  %r12
  popq  %rbx
  ret

  .section .rodata
.p2align 3
.Lone:
  .double 1.0
.Lln2:
  .double 0.69314718055994530942
.Linv_ln2:
  .double 1.4426950408889634074
.Lsigmoid_lower:
  .double -14.0
.Lsigmoid_upper:
  .double 14.0
.Ltanh_lower:
  .double -8.0
.Ltanh_upper:
  .double 8.0
.Lnegative_one:
  .double -1.0
.Lexp_c2:
  .double 0.5
.Lexp_c3:
  .double 0.16666666666666666667
.Lexp_c4:
  .double 0.041666666666666666667
.Lexp_c5:
  .double 0.0083333333333333333333
.Lexp_c6:
  .double 0.0013888888888888888889
.Lexp_c7:
  .double 0.00019841269841269841270
.Lexp_c8:
  .double 0.000024801587301587301587
.Lexp_c9:
  .double 0.0000027557319223985890653

  .section .note.GNU-stack,"",@progbits
