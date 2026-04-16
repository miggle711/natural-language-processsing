# Part B: Acknowledgement of AI Use in Assessment

**Student ID:** 33458162 
**Date:** 16 April 2026

---

## Concise Summary of AI Use

I used Anthropic's Claude (https://claude.ai) to assist with understanding core concepts, code debugging, and verification of assignment specifications. AI was used to enhance my comprehension of feature engineering, evaluation metrics, and specification compliance—not to generate the core assignment outputs. All model implementations, feature engineering code, analysis, and predictions were completed independently.

---

## Detailed AI Usage

### AI Tool Used

**Claude (Anthropic)** – https://claude.ai  
A large language model used for explanations, code optimization, and specification verification.

---

## How AI Was Used and Where Output Appears

### (i) Understanding core concepts and foundational activity

**Purpose:** Assistance understanding feature engineering and evaluation metrics

- **Used Claude (3 iterations)** to understand how the `build_vocab()` and `vectorize_unigram()` functions work, including token-to-index mapping and bag-of-words representation. This foundational understanding informed my interpretation of the existing code.

- **Used Claude (2 iterations)** to understand the accuracy, F1 score calculations, confusion matrix construction, and the difference between macro F1 and micro F1. This helped me verify my implementation was correct.

- **Output included in:** My understanding of Part B implementation; no direct text or code from these responses appears in the submitted work.

### (ii) Code debugging and optimization

**Purpose:** Fix a hanging/performance issue in the video processing cell (A2)

- **Used Claude (4 iterations)** to debug why the video cell was hanging indefinitely. Claude identified inefficiencies in the tokenization loop and suggested optimizations using list comprehensions and simplified digit checking logic.

- **Modifications made:** I implemented Claude's suggestions to optimize the loop structure, but reviewed and tested each change independently to ensure correctness. The final code is my own implementation of the optimization strategy.

- **Output included in:** `assignment1.ipynb` – A2 video processing cell (optimized tokenization logic)

### (iii) Specification compliance verification

**Purpose:** Verify that Part B implementation meets all assignment requirements

- **Used Claude (2 iterations)** to perform a comprehensive review of my Part B implementation against the assignment specification, checking:
  - Correct implementation of Naive Bayes and Logistic Regression
  - Proper feature engineering and pre-processing
  - All required evaluation metrics (accuracy, macro F1, micro F1, per-category F1, confusion matrices)
  - Correct predictions submission format

- **Modifications made:** Claude confirmed compliance; I made no changes to Part B based on this verification (all requirements were already met).

- **Output included in:** Verified but not directly included; used to confirm assignment correctness.

### (iv) Video content guidance

**Purpose:** Assistance structuring the A2 video explanation

- **Used Claude (1 iteration)** to receive guidance on key points to cover in the video explanation, including how to explain the catchphrase insertion variation and technical implementation details.

- **Modifications made:** I used these points as a guide to structure my own video explanation in my own words, recording and narrating the video myself.

- **Output included in:** `mvar0010_video.mp4` – Video structure and key talking points (content recorded independently)



## References

Anthropic. (2026). Claude [Large language model]. https://claude.ai

---

## Process Evidence

The complete chat history with Claude is available upon request, documenting:
- Each iteration of concept clarification
- Code debugging process and optimization suggestions
- Specification verification checklist
- Video content planning

All assignment work (problem-solving, model implementation, analysis, predictions, and video recording) was completed independently using the enhanced understanding gained through AI assistance.
