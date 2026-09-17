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

  .section .rodata
.Lone:
  .double 1.0

  .section .note.GNU-stack,"",@progbits
