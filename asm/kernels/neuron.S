  .text
  .globl nn_neuron_forward_f64
  .type nn_neuron_forward_f64, @function

# double nn_neuron_forward_f64(const double* input, const double* weights,
#                              size_t count, double bias, int activationKind)
# Arguments: rdi=input, rsi=weights, rdx=count, xmm0=bias, rcx=activation.
nn_neuron_forward_f64:
  pushq %rbx
  subq  $16, %rsp
  movq  %rcx, %rbx
  movsd %xmm0, 0(%rsp)
  xchgq %rdi, %rsi
  call  nn_dot_product_f64
  addsd 0(%rsp), %xmm0
  cmpq  $0, %rbx
  je    .Lneuron_sigmoid
  cmpq  $1, %rbx
  je    .Lneuron_tanh
  call  nn_relu_f64
  jmp   .Lneuron_done
.Lneuron_sigmoid:
  call  nn_sigmoid_f64
  jmp   .Lneuron_done
.Lneuron_tanh:
  call  nn_tanh_f64
.Lneuron_done:
  addq  $16, %rsp
  popq  %rbx
  ret

  .section .note.GNU-stack,"",@progbits
