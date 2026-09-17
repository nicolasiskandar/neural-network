  .text
  .globl nn_fast_layer_forward_f64
  .type nn_fast_layer_forward_f64, @function

# The seventh argument, activationKind, is passed on the stack under the
# System V AMD64 ABI. Built-in activation kinds are 0=sigmoid, 1=tanh, 2=ReLU.
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

  .section .note.GNU-stack,"",@progbits
