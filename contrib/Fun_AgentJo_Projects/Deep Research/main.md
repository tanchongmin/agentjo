# What is Memory?

## 1. Introduction
Memory is a fundamental process present in both biological and artificial systems. It encompasses the mechanisms through which information—whether temporal events, spatial features, or sensory inputs—is encoded, stored, and later retrieved [[1]](https://arxiv.org/abs/2412.15292), [[2]](https://arxiv.org/abs/q-bio/0403025), [[8]](https://arxiv.org/abs/2006.12616). In biological systems, memory is intimately tied to the adaptability of neural circuits and learning processes, whilst in artificial intelligence, it informs model architectures that must both store past experiences and dynamically update to meet new contexts [[9]](https://arxiv.org/abs/2403.01518). This report examines these multiple aspects of memory across various systems and proposes potential future research directions.

## 2. Types of Memory
Memory manifests in several forms and architectures:

- **Temporal and Scale Invariant Memory:** Deep reinforcement learning models deploy mechanisms, such as recurrent neural networks (RNNs) and Long Short-Term Memory networks (LSTMs), with innovations like scale invariant (or log-compressed) representations of time, facilitating robust performance across varying temporal conditions [[1]](https://arxiv.org/abs/2412.15292).

- **Synaptic Memory in Neural Networks:** Biological systems model memory through synaptic plasticity. Processes like long-term potentiation (LTP) and long-term depression (LTD) enable networks to continuously reshape stored information, offering both stability for frequently used patterns and flexibility through synaptic weakening [[2]](https://arxiv.org/abs/q-bio/0403025), [[7]](https://arxiv.org/abs/0905.2125).

- **Volatile vs. Non-volatile Memory:** In self-organizing systems, memory can be volatile—requiring ongoing activity to maintain—or non-volatile, where information is embedded in the network connectivity even after removal of sustained activation [[5]](https://arxiv.org/abs/2303.12225).

- **Distributed Memory in Language Models:** Modern language models utilize two kinds of memory: one stored in the model's weights (long-term storage) and another in the transient activation states (context memory), each contributing differently to model performance [[9]](https://arxiv.org/abs/2403.01518).

- **Episodic and Integrated Memory:** In memory-augmented architectures, distinct episodic memories store separate, pattern-separated experiences which can later be integrated during inferential reasoning [[10]](https://arxiv.org/abs/2001.10913), as well as frameworks that use saliency-augmented memory for continual learning to prevent catastrophic forgetting [[6]](https://arxiv.org/abs/2212.13242).

## 3. How Memory can be Adaptive
Adaptivity is central to effective memory systems:

- **Scale Invariance:** In computational models, memory is made adaptive by ensuring that the representation of temporal history shifts proportionally when the time scale is altered, maintaining functionality across different environments without the need for readjustment [[1]](https://arxiv.org/abs/2412.15292).

- **Synaptic Plasticity and Selective Updating:** Biological networks adapt by modulating synaptic strengths; pathways responsible for errors are downregulated, while those that yield successful responses are reinforced, allowing quick reconfiguration in response to new information [[2]](https://arxiv.org/abs/q-bio/0403025), [[7]](https://arxiv.org/abs/0905.2125).

- **Dynamic Adaptation in Oscillator Networks:** Memory associated with potential landscapes can shift in response to external forcing, enabling the network to adapt its stored states dynamically to accommodate new patterns [[3]](https://arxiv.org/abs/2008.07448).

- **Online and Continuous Adaptation:** Approaches like dynamic evaluation enable large language models to update their weight memory during inference, thus effectively adapting to distribution shifts and extending their working context [[9]](https://arxiv.org/abs/2403.01518).

## 4. How Memory Schema is Created
Memory schema creation is a multi-step process across different systems:

- **Computational Transformations:** In deep learning, particularly for temporal data, schemas can be derived using mathematical tools such as the Laplace and inverse Laplace transform. This process builds a compressed yet sequential representation of past inputs, mirroring time cells observed in the mammalian brain [[1]](https://arxiv.org/abs/2412.15292).

- **Synaptic Organization:** In biological and bio-inspired networks, repetitive exposure to stimuli leads to the self-organization of neural circuits where bottom-up sensory information, lateral connections, and top-down feedback converge to form stable memory representations [[2]](https://arxiv.org/abs/q-bio/0403025), [[7]](https://arxiv.org/abs/0905.2125).

- **Potential Landscapes:** In oscillator networks, memory schemas emerge as valleys in amplitude or phase potential landscapes derived from differential equations. These landscapes offer stable attractors corresponding to remembered states, which can be recalled through associative dynamics [[3]](https://arxiv.org/abs/2008.07448).

- **Sparse and Salient Representations:** In continual learning models, schemas are constructed by filtering input data through saliency maps, retaining only the most informative features. These features are stored sparsely, and later reconstructed using inpainting methods to alleviate memory storage issues and address catastrophic forgetting [[6]](https://arxiv.org/abs/2212.13242).

- **Evolutionary Processes:** Evolutionary mechanisms enable the formation of memory in recurring processes, where neuroevolution helps shape networks that can sustain and recall short-term memories across multiple time steps [[4]](https://arxiv.org/abs/1204.3221).

## 5. Future Focus Areas on Memory
Future research on memory can be directed along several innovative paths:

- **Integration of Dynamic Discounting:** Investigating the integration of scale invariant temporal discounting with adaptive memory architectures to improve learning efficiency and reduce the need for hyperparameter tuning [[1]](https://arxiv.org/abs/2412.15292).

- **Optimizing Synaptic Balance:** Further exploration of methods to balance synaptic plasticity (LTP and LTD) to prevent issues like runaway potentiation, ensuring both adaptability and long-term stability in biological and artificial systems [[2]](https://arxiv.org/abs/q-bio/0403025).

- **Advanced Neuromorphic Systems:** Extending current methods by exploring global feedback, synchrony in neural oscillator models, and the impact of noise on memory stability could bridge the gap between theoretical models and practical neuromorphic applications [[3]](https://arxiv.org/abs/2008.07448), [[5]](https://arxiv.org/abs/2303.12225).

- **Enhanced Online Adaptation:** With large language models becoming ubiquitous, developing more efficient strategies for online adaptation while balancing computational costs and memory efficiency represents a key research challenge [[9]](https://arxiv.org/abs/2403.01518).

- **Scalable Memory Architectures:** For memory-augmented networks, future work should aim at scalable architectures that can handle longer sequences, deeper relationships, and multi-modal data while ensuring rapid retrieval and minimal interference [[10]](https://arxiv.org/abs/2001.10913), [[6]](https://arxiv.org/abs/2212.13242).

## 6. Conclusion
Memory is a dynamic, multifaceted attribute crucial for both biological cognition and artificial intelligence. From synaptic plasticity in biological systems to advanced computational schemas involving Laplace transforms and dynamic evaluation, memory enables systems to store, retrieve, and adapt information for improved decision-making and learning. The convergence of insights from neuroscience, dynamical systems, and machine learning suggests that future advances in memory research could lead to AI systems capable of continual learning and robust performance in changing environments.

---

### List of Sources

[1]: Banerjee, K. et al. (2024). Deep reinforcement learning with time-scale invariant memory. Retrieved from https://arxiv.org/abs/2412.15292

[2]: Wakeling, J. R. (2004). Adaptivity and `Per learning. Retrieved from https://arxiv.org/abs/q-bio/0403025

[3]: Hoppensteadt, F. (2020). A Frequency-Phase Potential for a Forced STNO Network: an Example of Evoked Memory. Retrieved from https://arxiv.org/abs/2008.07448

[4]: Lakhman, K., & Burtsev, M. (2012). Neuroevolution Results in Emergence of Short-Term Memory for Goal-Directed Behavior. Retrieved from https://arxiv.org/abs/1204.3221

[5]: Neves, F. S., & Timme, M. (2023). Volatile Memory Motifs: Minimal Spiking Neural Networks. Retrieved from https://arxiv.org/abs/2303.12225

[6]: Bai, G., Ling, C., Gao, Y., & Zhao, L. (2022). Saliency-Augmented Memory Completion for Continual Learning. Retrieved from https://arxiv.org/abs/2212.13242

[7]: Jitsev, J., & von der Malsburg, C. (2010). Experience-driven formation of parts-based representations in a model of layered visual memory. Retrieved from https://arxiv.org/abs/0905.2125

[8]: Schillaci, G., Miranda, L., & Schmidt, U. (2020). Prediction error-driven memory consolidation for continual learning. Retrieved from https://arxiv.org/abs/2006.12616

[9]: Rannen-Triki, A., Bornschein, J., Pascanu, R., Hutter, M., György, A., Galashov, A., Teh, Y. W., & Titsias, M. K. (2024). Revisiting Dynamic Evaluation: Online Adaptation for Large Language Models. Retrieved from https://arxiv.org/abs/2403.01518

[10]: Banino, A., Puigdomènech Badia, A., Köster, R., Chadwick, M. J., Zambaldi, V., Hassabis, D., Barry, C., Botvinick, M., Kumaran, D., & Blundell, C. (2020). MEMO: A Deep Network for Flexible Combination of Episodic Memories. Retrieved from https://arxiv.org/abs/2001.10913