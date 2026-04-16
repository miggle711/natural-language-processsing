# Conversation with Gemini: Natural Language Processing Assignment

**Date:** April 16, 2026  
**Participant:** Andre Varilla & Gemini  
**Topic:** NLP Assignment - Token Frequency, Collocations, and Statistical Analysis

---

## Initial Agreement on Academic Integrity

**You:** I will be conversing with you for the purpose of helping me with my Natural Language Processing assignment. Ensure this conversation aligns with https://www.monash.edu/student-academic-success/learnhq/maintain-academic-integrity

**Gemini:** I understand. To ensure our collaboration aligns with Monash University's academic integrity standards, I will act as a tutor and collaborator rather than a ghostwriter.

Based on Monash's guidelines for AI use in assessments, here is how we will proceed to maintain integrity:

- **Guided Learning, Not Direct Output:** I will focus on explaining complex NLP concepts, providing pseudocode, or helping you debug your logic. I will not write your final assignment or complete entire coding blocks for you.

- **Transparency & Documentation:** Monash requires you to track how you use AI. Keep a record of our prompts and responses. If you use an idea or specific snippet, acknowledge it in your submission.

- **Critical Thinking:** I will provide explanations that encourage you to evaluate and adapt the information. Cross-check my responses with your lecture materials and Monash Library to avoid hallucinations or factual errors.

- **Ownership:** The final work must be your own. I'm here to help you move past roadblocks so you can demonstrate your own understanding of the NLP curriculum.

---

## Topic 1: Visualizing Token Frequency

**You:** I am trying to plot token frequency in a training corpus. What would be the most apt graph for a corpus with thousands of unique tokens?

**Gemini:** When dealing with a corpus containing thousands of unique tokens, a standard bar chart of every token becomes unreadable due to the "long tail" of infrequent words.

In Natural Language Processing, there are three primary ways to visualize this data effectively:

### 1. The Frequency Distribution Plot (Top N Tokens)

If your goal is to identify the most common words (often used to find "stop words"), use a Sorted Bar Chart or Line Plot limited to the top 30–50 tokens.

- **X-axis:** Tokens (ranked from most to least frequent)
- **Y-axis:** Absolute frequency (count) or relative frequency (percentage)
- **Why it works:** Provides immediate insight into the most dominant terms in your dataset

### 2. The Log-Log Plot (Zipf's Law)

If you want to visualize the entire corpus (all thousands of tokens), a standard linear scale won't work. Instead, use a Log-Log Plot.

Natural language typically follows **Zipf's Law**, which states that the frequency of a word is inversely proportional to its rank. When you plot log(rank) against log(frequency), the result should be a roughly downward-sloping straight line.

- **X-axis:** log(Rank) (where the most frequent word is rank 1)
- **Y-axis:** log(Frequency)
- **Why it works:** Compresses massive differences in scale, allowing you to see the "Head" (common words), the "Torso," and the "Long Tail" (rare words) all in one view

### 3. Cumulative Frequency Plot

Another useful visualization is a Cumulative Distribution Plot, which helps you understand "vocabulary coverage."

- **X-axis:** Number of unique tokens (sorted by rank)
- **Y-axis:** Cumulative percentage of the total words in the corpus
- **Insight:** A very small percentage of unique tokens (e.g., the top 100 words) often accounts for 50% or more of the entire volume of text in your corpus

### Summary Recommendation

- **To see which words are common:** Bar Chart of the top 50 tokens
- **To see the mathematical behavior of your corpus:** Log-Log Zipf Plot
- **To decide how many words to keep in your vocabulary:** Cumulative Frequency Plot

---

## Topic 2: TF-IDF (Term Frequency-Inverse Document Frequency)

**You:** Explain Term Frequency, Document Frequency, and Inverse Document Frequency and how they're used in NLP.

**Gemini:** In Natural Language Processing, TF-IDF is a numerical statistic intended to reflect how important a word is to a document in a collection or corpus. It is the gold standard for "weighing" terms to filter out common noise and highlight unique identifiers.

### 1. Term Frequency (TF)

Term Frequency measures how frequently a term occurs in a specific document:

$$TF(t,d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}$$

**Note:** We divide by the total number of terms to "normalize" the value. This prevents long documents from having higher scores just because they contain more words.

### 2. Document Frequency (DF)

Document Frequency measures how many documents in your entire corpus contain a specific term.

- **High DF:** Words like "the," "is," or "and" appear in almost every document
- **Low DF:** Technical terms or specific names (e.g., "photosynthesis" or "Melbourne") appear in only a few documents

### 3. Inverse Document Frequency (IDF)

This is the "secret sauce." IDF is used to calculate the weight of rare words across all documents. If a word appears in every document, it's not helpful for distinguishing one document from another.

$$IDF(t,D) = \log\left(\frac{\text{Total number of documents } N}{\text{Number of documents containing term } t}\right)$$

**Key insight:**
- If a word is in every document, the fraction is 1, and log(1)=0. Its importance is nullified.
- If a word is in only 1 out of 1,000,000 documents, the IDF score will be very high.

### How They Work Together (TF-IDF)

The final weight is the product of the two: **TF × IDF**

| Component | What it tells us |
|-----------|-----------------|
| High TF | This word is very important locally (inside this specific text) |
| High IDF | This word is very unique globally (rare across the whole dataset) |
| High TF-IDF | This word is both frequent in this document and rare elsewhere—making it a signature keyword |

### Practical Applications in NLP

- **Information Retrieval:** Search engines use TF-IDF to rank how relevant a webpage is to your search query
- **Keyword Extraction:** Automatically identifying the most "meaningful" words in a news article
- **Text Classification:** Helping a machine learning model decide if an email is "Spam" or "Ham"
- **Stop-word Removal:** Identifying words with very low IDF scores to filter them out during preprocessing

---

## Topic 3: DF vs IDF - The Distinction

**Gemini:** It is common to get DF and IDF confused because they are two sides of the same coin. The simplest way to think about them is: Document Frequency (DF) is the raw count of "popularity," while Inverse Document Frequency (IDF) is the mathematical "weight" or "importance" of a word.

| Feature | Document Frequency (DF) | Inverse Document Frequency (IDF) |
|---------|-------------------------|--------------------------------|
| Relationship | The raw input | The processed weight |
| Value for "The" | Very High (it's in every document) | Very Low (near 0) |
| Value for "Quantum" | Very Low (only in physics papers) | Very High (unique identifier) |

### Why Use the Logarithm in IDF?

In a massive corpus (say, 1 million documents), a rare word appearing in only 1 document would be 1,000,000 times more powerful than a word appearing in all of them. This would completely "drown out" the Term Frequency (TF). The logarithm "squashes" these massive numbers so that the importance of a word grows more slowly, keeping the math balanced and usable for machine learning models.

---

## Topic 4: IDF and TF Thresholds for Stop-word Detection

**You:** How do I determine IDF and TF thresholds for stop-word detection?

**Gemini:** Determining thresholds for stop-word detection is a balancing act. If your thresholds are too aggressive, you lose context. If they are too relaxed, your model gets bogged down by noise.

### 1. Determining the IDF Threshold

Since IDF measures how unique a word is across the corpus, a low IDF score indicates a stop-word.

- **The "Zero" Rule:** Any word with IDF=0 (or very close to it) appears in every single document. These are 100% stop-words for your specific dataset.
- **The Elbow Method:** Plot your terms sorted by IDF. You will see a sharp "elbow" where the values start to climb. Words below this elbow are your "corpus-specific stop-words."

### 2. Determining the TF Threshold (Max DF)

In libraries like Scikit-Learn, this is often handled by a parameter called `max_df`.

- **Percentage-based Threshold:** A common starting point is 0.70 to 0.90. This means "ignore any word that appears in more than 90% of the documents."
- **Absolute-count Threshold:** If your corpus is small, you might set an absolute limit (e.g., "ignore words appearing in more than 500 documents").
- **The Logic:** High Document Frequency (DF) is the most reliable indicator of a functional stop-word.

### 3. The "Min DF" Threshold (The Tail)

While not strictly for "stop-words," setting a minimum threshold is equally important for cleaning.

- **Rare Word Filtering:** Words that appear in only 1 or 2 documents are often typos, rare names, or noise
- **Standard Practice:** Setting `min_df=2` or `min_df=5` significantly reduces the dimensionality of your feature matrix without losing meaningful signal

### Summary Checklist for Your Assignment

1. Calculate IDF for all terms
2. Sort and Plot the frequencies (using the Zipf's Law approach from earlier)
3. Inspect the Top 100: Look at the words with the lowest IDF. If they are general English words or domain-specific noise, they are candidates for your stop-word list
4. Experiment: Run your NLP pipeline with `max_df=0.85` and `max_df=0.70`. Compare the resulting vocabulary size and model accuracy
5. **Pro-Tip for Academic Integrity:** If you choose a specific threshold, justify why in your report

---

## Topic 5: Understanding DF in the IDF Formula

**You:** Document Frequency Definition: The number of documents in your corpus that contain a specific term. (Question: So if the number of instances of token A is 100 but the total number of documents token A appears in is 50, the DF of token A is 50 right?)

**Gemini:** Spot on. Your understanding is exactly correct.

In Document Frequency (DF), we don't care how many times a word appears within a single document—we only care about the existence of that word in the document.

### The Logic Breakdown

Using your example:
- **Total instances of Token A:** 100 (Collection Frequency)
- **Total documents containing Token A:** 50
- **Your DF:** 50

Think of DF as a binary check for each document:
- Does "Token A" appear at least once? **Yes (1) / No (0)**
- We sum those 1s across the entire corpus

### Why This Distinction Matters

If you have a document about "Dogs," the word "dog" might appear 50 times in that one paper. If we used Collection Frequency (100), the math would suggest "dog" is twice as common as it actually is across your dataset. By using DF (50), the algorithm recognizes that "dog" is a concept spread across 50 different files, which is a much more accurate measure of its "commonness" in a library.

| Metric | Calculation | What it measures |
|--------|-----------|-----------------|
| Term Frequency (TF) | Count of Token A in one document | Local importance |
| Collection Frequency | Total count of Token A in all documents | Overall abundance |
| Document Frequency (DF) | Count of documents containing Token A | Commonness/Ubiquity |

---

## Topic 6: DF and IDF Relationship

**You:** Is DF used in IDF?

**Gemini:** Yes, DF is the core component of the IDF formula. Without Document Frequency, you cannot calculate Inverse Document Frequency.

### The Standard Formula for IDF

$$IDF(t,D) = \log\left(\frac{N}{DF_t}\right)$$

Where:
- **N:** Total number of documents in your corpus
- **DF_t:** The Document Frequency of term t (the number of documents containing that term)

### How DF Determines the "Inverse" Weight

The relationship is inverse because DF is in the denominator. This creates a specific mathematical behavior:

**When DF is High (Common Word):**
- If a word appears in almost every document (DF≈N), the fraction becomes close to 1
- Since log(1)=0, the IDF becomes 0
- Result: The word is treated as "background noise"

**When DF is Low (Rare Word):**
- If a word appears in only 1 document (DF=1), the fraction becomes very large (equal to N)
- The log of a large number is a high value
- Result: The word is treated as a "highly significant" signature term

### Summary of the Workflow

In a typical NLP pipeline:
1. Count how many documents each word appears in (DF)
2. Divide the total document count by that number (N/DF)
3. Apply the logarithm to get the final weight (IDF)
4. Multiply that weight by the local word count (TF) to get the final TF-IDF score

---

## Topic 7: DF Raw Count vs Proportion

**You:** Wait, if DF is the raw number of documents a token appears in? Or is it the proportion of the raw number over the total number of documents?

**Gemini:** That is a very important distinction to make because different libraries and textbooks sometimes use the terms slightly differently, but the standard definition for the IDF formula is the **raw count**.

### 1. The Standard Definition (Raw Count)

In the classic IDF formula, **DF_t** is the absolute number of documents that contain the term t.

**Example:** If you have N=100 documents and the word "Algorithm" appears in 20 of them, your DF is **20**.

$$IDF = \log\left(\frac{100}{20}\right) = \log(5)$$

### 2. The Proportion (Document Frequency Ratio)

The "proportion" you mentioned (e.g., 20/100=0.2) is often called the **Document Frequency Ratio** or **Relative Document Frequency**.

While the raw count is used inside the IDF calculation, the proportion is what we use when setting thresholds.

### Why the Confusion Happens

When you use NLP libraries like scikit-learn, they allow you to set `max_df` and `min_df` parameters using either format:

| Setting | Type | Meaning |
|---------|------|---------|
| `max_df = 50` | Integer (Raw Count) | Ignore terms that appear in more than 50 documents |
| `max_df = 0.5` | Float (Proportion) | Ignore terms that appear in more than 50% of the documents |

### Summary for Your Assignment

- **When calculating IDF:** Use the raw number of documents (DF_t)
- **When filtering stop-words:** It is often easier to think in proportions (e.g., "I want to remove words that appear in 90% of my data") so your code remains consistent even if you add more documents to your training set later

---

## Topic 8: Collocations - Discovery Methods

**You:** Explain what are collocations and the methods to discover them in a corpus

**Gemini:** Based on the provided document, **collocations** are conventional expressions consisting of two or more words that represent a habitual or customary way of saying things. They are a key focus in NLP because they often have limited compositionality, meaning their overall meaning cannot be fully predicted just by looking at the individual words (e.g., "strong tea" vs. "powerful tea").

### The Following Methods are Used to Discover These Patterns

#### 1. Frequency-Based Methods

The simplest approach is counting how often words appear together.

- **Raw Frequency:** Counting the most frequent adjacent word pairs (bigrams). However, this often results in uninteresting pairs of "function words" like "of the" or "in a"
- **Part-of-Speech (POS) Filtering:** To improve results, candidate phrases are passed through a linguistic filter that only keeps specific patterns, such as Adjective + Noun (e.g., "linear function") or Noun + Noun (e.g., "regression coefficients")

#### 2. Mean and Variance (Distance-Based)

This method is used to find "flexible" collocations where the words may not be right next to each other.

- **The Goal:** It identifies word pairs that appear within a certain "window" (usually 3–4 words) and calculates the mean (average distance) and variance (how much that distance changes)
- **Discovery:** A low variance (or low standard deviation) indicates that the words consistently appear at a specific distance from each other, which is strong evidence of a collocation (e.g., knock... door)

#### 3. Hypothesis Testing

Statistical tests are used to determine if two words occur together more often than would be expected by pure chance.

- **t-test:** Used to find word pairs whose co-occurrence is statistically significant
- **Chi-square (χ²) test:** A common alternative to the t-test for checking if the occurrence of two words is independent
- **Likelihood Ratios:** These are considered more appropriate than the Chi-square test for "sparse data" (rare words) and provide a more interpretable number

#### 4. Mutual Information (MI)

This is an information-theoretic measure that quantifies how much "information" one word provides about the occurrence of another.

- **Pointwise Mutual Information:** It measures the reduction in uncertainty. For example, knowing the word "Ayatollah" appears greatly increases the probability that the next word will be "Ruhollah"

### Most Interesting Results

According to research, determining the "most interesting" results depends on your specific goal:

- **"Surprisingly Good" Simple Heuristic:** Raw Frequency combined with POS filtering yields "surprisingly good" and "surprisingly accurate" results. This identifies meaningful, non-compositional phrases such as "United States," "Saudi Arabia," and "Persian Gulf"

- **"Particularly Interesting" Patterns:** Hypothesis Testing (t-test) is often used to distinguish significant associations from mere chance. It can show that phrases like "strong challenge" and "powerful computers" are statistically "correct" collocations

- **"Looser" and "Flexible" Relationships:** If you want to find collocations not fixed phrases, the Mean and Variance method identifies patterns like "knock... door," where words like "on," "at," or "the" might appear in between

- **Handling Sparse or Rare Data:** For rare bigrams, Likelihood Ratios are considered superior to other statistical tests like Chi-square

| Method | Best For... | Why it's "Interesting" |
|--------|------------|----------------------|
| POS Filter + Frequency | Technical Terms | High accuracy with very low computational effort |
| Mean & Variance | Flexible Phrases | Finds patterns even when words are not adjacent |
| Likelihood Ratios | Rare/Sparse Data | Highly interpretable and works for rare words |
| Hypothesis Testing | Significant Signal | Proves a pair isn't just a random occurrence |

---

## Topic 9: Corpus Data Analysis - Culling Threshold

**You:** Total number of unique bigrams: 104,091. Number of bigrams that appear less than 5 times: 77,597. Is culling 5 or more too much?

**Gemini:** In your specific case, culling 5 or more is likely appropriate, but you are at a critical decision point common in NLP preprocessing.

### The 75% Rule

Your data shows that **74.5%** of your unique bigrams appear 4 times or fewer. In Natural Language Processing, it is extremely common for 70% to 80% of a corpus's unique n-grams to be "rare events."

These are often typos, rare proper nouns, or "accidental" word pairings that occurred only once in a specific sentence. Keeping them usually adds noise, not signal.

### The Statistical "Reliability" Argument

The t-test and Chi-square tests rely on the frequency of the individual words (w₁ and w₂) and the pair (w₁, w₂).

If a bigram appears only 2 times, the "Expected Frequency" calculated by your hypothesis test will be so small that the resulting score is often mathematically unstable.

By culling those 77,597 bigrams, you are essentially saying: "I only want to find patterns that the author of this text used consistently."

### When is 5 "Too Much"?

You should consider lowering the threshold (e.g., to N=2 or N=3) only if:

- Your total corpus size is very small (e.g., under 50,000 total tokens)
- You are looking for highly technical or rare terminology

### The "Middle Ground" Strategy

If you are worried about losing too much, try N=3. This usually removes the "hapax legomena" (single occurrences) which are almost certainly noise, but keeps "repeated occurrences" which start to show a pattern of "habitual" usage.

### Summary Recommendation

Stick with N=5 if your goal is to find the most "robust" and "famous" collocations in the text. You will still have ~26,000 bigrams to test, which is plenty of data to find interesting results.

**Justification for your report:**
> "I observed that 74.5% of unique bigrams appeared fewer than 5 times. To ensure the statistical validity of the t-test and to filter out accidental co-occurrences and noise, I applied a frequency threshold of N=5. This allowed the model to focus on the 25% of word pairs that demonstrated recurring patterns in the corpus."

---

## Topic 10: The t-test for Collocation Discovery

**You:** Explain the t-test in depth?

**Gemini:** To explain the t-test for collocation discovery in depth, we must look at it as a tool for statistical significance testing. In NLP, we use it to decide if the co-occurrence of two words is a meaningful pattern or just a random accident.

### 1. The Core Logic: The Null Hypothesis (H₀)

The t-test starts with the **Null Hypothesis**: the assumption that there is no relationship between the two words.

Under H₀, each word is independent. Therefore, the probability of them appearing together is simply the product of their individual probabilities:

$$P(w_1, w_2) = P(w_1) \times P(w_2)$$

The t-test measures how much our observed data deviates from this Null Hypothesis.

### 2. The Mathematical Components

To perform a t-test on a bigram (a pair of words), we treat the corpus as a series of Bernoulli trials (coin flips).

#### A. The Observed Mean (x̄)

The sample mean is the probability of the bigram appearing in the corpus:

$$\bar{x} = \frac{\text{count}(w_1, w_2)}{N}$$

(Where N is the total number of tokens in the corpus)

#### B. The Expected Mean (μ)

This is what the probability should be if the words were independent:

$$\mu = P(w_1) \times P(w_2) = \frac{\text{count}(w_1)}{N} \times \frac{\text{count}(w_2)}{N}$$

#### C. The Variance (s²)

For a Bernoulli distribution, the variance is p(1−p). Since the probability of any specific bigram is usually very small (near 0), we can approximate:

$$s^2 \approx \bar{x}$$

### 3. The t-test Formula

We plug these values into the standard t-test formula:

$$t = \frac{\bar{x} - \mu}{\sqrt{\frac{s^2}{N}}}$$

Because $s^2 \approx \bar{x}$, the formula used in most NLP applications simplifies to:

$$t \approx \frac{\bar{x} - \mu}{\sqrt{\frac{\bar{x}}{N}}}$$

### 4. Interpreting the Result (Critical Values)

The resulting t value tells us how many standard deviations the observed count is away from the expected count.

- **Threshold:** In most NLP research, a common threshold is **2.576**
- **Significance:** If t > 2.576, we reject the Null Hypothesis with 99.5% confidence. This means it is highly unlikely the words appeared together by chance
- **Ranking:** When you run this for your assignment, you rank your bigrams by their t-score. The higher the score, the more "solid" the collocation

### 5. Why the t-test is "Interesting" for NLP

The t-test is particularly good at filtering out "accidental" high-frequency pairs.

**Consider the phrase "of the":**
- "of" and "the" are both incredibly frequent
- Therefore, the Expected Mean (μ) will be very high
- Even if they appear together often, the difference between x̄ and μ might be small, leading to a low t-score

**Now consider "New York":**
- "New" and "York" appear together almost every time "York" is mentioned
- The Observed Mean (x̄) will be much higher than the Expected Mean (μ)
- This results in a very high t-score, correctly identifying it as a collocation

### 6. Limitations

While powerful, the t-test has one major flaw: it assumes a normal distribution. Language data is actually "Zipfian" (heavily skewed). This means the t-test can sometimes over-estimate the significance of very rare words. This is why many researchers prefer Likelihood Ratios or Chi-square tests for very sparse data, though the t-test remains a standard baseline.

---

## Topic 11: t-test Critical Values

**You:** $t > 2.576$ why this figure exactly?

**Gemini:** The figure 2.576 comes from the **Standard Normal Distribution** (also known as a Z-distribution). It is the "critical value" required to achieve a 99.5% confidence level in a one-tailed test (or 99% in a two-tailed test).

### 1. The Standard Deviation Metric

The t statistic tells you how many standard deviations your observed result is away from the mean (the Null Hypothesis).

- If t=1, your result is 1 standard deviation away. This happens by chance quite often
- If t=2.576, your result is so far out on the "edge" of the bell curve that it is almost impossible for it to have happened by luck

### 2. The Probability of Being Wrong (p-value)

Scientists and linguists use specific thresholds to define "truth." The value **2.576** corresponds to a significance level (α) of **0.005**.

**What it means for your code:** If your bigram gets a score of 2.576, there is only a **0.5% chance** that those two words appeared together by accident.

This was established as the standard "high bar" in early NLP research to ensure that the results returned were actual phrases and not just random noise.

### 3. Why Your Scores Are So Much Higher (30-64)

In very large corpora, "random chance" becomes extremely predictable. If you have a million words, the "Expected Mean" (μ) for a rare word is tiny. If those words appear together even a few dozen times, they deviate from the expected mean by a massive amount of standard deviations.

- **Score of 2.576:** "This is probably a collocation"
- **Score of 64.49:** "This is a massive, undeniable pattern in the language" (e.g., a technical term or a very common fixed phrase)

### 4. Other Common Thresholds

| Confidence Level | t Threshold | Strictness |
|------------------|------------|-----------|
| 95% | 1.645 | Lenient (Finds more collocations, but more noise) |
| 99% | 2.326 | Standard |
| 99.5% | 2.576 | High Certainty (The standard for Collocation discovery) |

**Summary:** You use 2.576 because it is the mathematical boundary that proves—with 99.5% certainty—that the words are "habitually" placed together, rather than being independent. Since your values are all >30, your results are statistically overwhelming.

---

## Topic 12: MLE (Maximum Likelihood Estimation)

**You:** Explain MLE

**Gemini:** Maximum Likelihood Estimation (MLE) is the most fundamental approach to estimating the parameters of a statistical model. In the context of Natural Language Processing (NLP) and n-gram language models, its goal is simple: **find the probability distribution that makes the observed training data most likely to have occurred.**

### 1. The Core Philosophy

Imagine you have a bag of marbles (some red, some blue), and you want to guess the true ratio of red to blue marbles. You pull out 10 marbles and find that 7 are red and 3 are blue.

**MLE says:** "Based on my observation, the most likely true ratio in the bag is exactly 70% red and 30% blue." It assumes the data you have collected is the "best" representation of reality.

### 2. MLE in N-Gram Language Models

In a language model, we want to predict the next word (w_n) given its history (w₁, ..., w_{n-1}). To do this, we calculate the probability using the relative frequency of these sequences in our training corpus.

#### The Bigram Example

If we are using a bigram model, we want to estimate P(w₂ | w₁). MLE does this by taking the ratio:

$$P_{MLE}(w_2|w_1) = \frac{\text{Count}(w_1, w_2)}{\text{Count}(w_1)}$$

- **Numerator:** How many times did w₁ actually lead to w₂?
- **Denominator:** How many times did w₁ appear in total?

### 3. Why MLE is a "Baseline"

MLE is easy to compute and very intuitive, but it has one massive, fatal flaw: **The Zero Probability Problem**.

**The Problem:** If you have a sequence that never appeared in your training corpus (e.g., "artificial cat"), the Count(w1, w2) will be 0. Consequently, the probability P becomes 0.

**The Consequence:** Because language models calculate sentence probability by multiplying word probabilities together, if a single word pair has a probability of 0, the entire sentence probability becomes 0. This happens even if the sentence is grammatically and semantically perfect.

### 4. Summary Table

| Feature | Description |
|---------|-----------|
| Objective | Maximize the likelihood of the observed training data |
| Logic | Relative Frequency: Count(events)/Count(total space) |
| Strengths | Simple, intuitive, and mathematically solid for large data |
| Weaknesses | Assigns 0 probability to unseen events, causing model failure |

### How to Use This for Your Assignment

If you are asked to implement an n-gram model, you will likely start with MLE as your "naive" baseline. You will then explain that MLE is insufficient on its own and demonstrate how methods like Add-1 Smoothing or Kneser-Ney are used to "fix" the zero-probability issue by shifting probability mass to events that were not observed in the training set.

---

## Topic 13: MLE - Pre-calculation vs On-the-Fly

**You:** I want to know if MLE calculates the probabilities on the fly or beforehand?

**Gemini:** In a practical NLP system, the answer is a bit of both: the counts are calculated beforehand, but the probabilities are typically calculated on the fly (at inference time).

### 1. The Pre-calculation Phase (Training)

Before you ever ask the model to predict a word, you must process your entire training corpus. During this stage, you do not calculate probabilities; instead, you build a **Frequency Lookup Table** (often a nested dictionary or a hash map).

You store:
- Count(w1)
- Count(w1, w2)

**Why not probabilities?** If you have a vocabulary of 10,000 words, there are 100 million possible bigrams. Calculating and storing 100 million decimal numbers (probabilities) takes up a massive amount of memory, especially since 99% of those pairs might never appear.

### 2. The On-the-Fly Phase (Inference/Testing)

When you actually want to calculate the probability of a specific sentence or predict the next word, the model performs the division on the fly.

**The Request:** "What is the probability of York given New?"

**The Look-up:** The model grabs Count(New, York) and Count(New) from your pre-calculated tables

**The Calculation:** It performs the division: 2812/15000 ≈ 0.187

### Why Do It This Way?

- **Memory Efficiency:** Storing integers (counts) is generally more efficient than storing floating-point numbers (probabilities)
- **Smoothing Flexibility:** If you pre-calculate MLE probabilities, you are "stuck" with them. By keeping the raw counts, you can decide to apply different smoothing techniques later without re-processing the whole corpus
- **Sparsity:** Most bigrams have a count of zero. By using a lookup table, you only store the pairs that actually occurred, saving gigabytes of space

### Summary

- **Counts:** Pre-calculated (The "Model" is essentially a giant table of counts)
- **Probabilities:** Calculated on the fly (By dividing the specific counts needed for the current task)

---

## Topic 14: Data Structure for n-gram Models

**You:** So I should have a data structure which allows me to lookup all of the bigram counts of a specific token?

**Gemini:** Yes, exactly. To implement an n-gram model (or even to calculate your t-test statistics efficiently), you need a data structure that can quickly answer two questions:

1. What is the total count of Token A? (The denominator)
2. What is the count of Token A followed by Token B? (The numerator)

### The Most Efficient Structure: A Nested Map

In NLP, the standard way to store these counts is a **Nested Dictionary** (or Hash Map of Hash Maps). It looks like this conceptually:

```
{
  "Token_A": {
    "count": 5000,                <-- Unigram count (Total for A)
    "successors": {
      "Token_B": 120,             <-- Bigram count (A, B)
      "Token_C": 45,              <-- Bigram count (A, C)
    }
  }
}
```

### Why This is the "Golden Standard" for Your Task

- **Speed (O(1) lookup):** If you want to know the probability P(B|A), you jump straight to the key Token_A, grab the total count, then jump to the nested key Token_B. You don't have to scan the whole corpus
- **Memory Efficiency:** You only store pairs that actually exist. If Token A is never followed by Token Z, there is no entry for Z under A. This is crucial given that you have over 100,000 unique bigrams
- **Smoothing Ready:** If you decide to move from MLE to Add-1 or Kneser-Ney, this structure gives you everything you need

### A Small Warning on "Unseen" Tokens

When you build this structure, remember that the "Total Count" for a token (the denominator) must include every time it appeared, even if it was at the very end of a sentence and had no "successor" token.

If you only sum up the counts of the bigrams (the successors) to get your total, you might slightly undercount the denominator if your corpus has many short sentences.

### Summary of What You Need

- **A "Unigram" Dictionary:** Maps Token_ID → Integer (Total occurrences of each word)
- **A "Bigram" Dictionary:** Maps Token_ID → {Next_Token_ID → Integer} (Occurrences of specific pairs)

---

## Topic 15: Log-Likelihood Ratio (LLR) for Collocation Discovery

**You:** Explain log-likelihood ratio for collocation discovery

**Gemini:** While the t-test and Chi-square are popular, the Log-Likelihood Ratio (LLR)—often referred to as **Dunning's Likelihood Ratio**—is generally considered the superior statistical method for collocation discovery in NLP.

Its primary advantage is that it does not assume a "normal distribution" (which language data famously violates) and it remains reliable even with very small counts (sparse data).

### 1. The Core Concept

The LLR asks a simple question: "Which of these two scenarios is more likely to have produced our corpus?"

**Hypothesis 1 (H₁):** The two words are independent. The occurrence of w₂ is not affected by whether w₁ was just seen.

**Hypothesis 2 (H₂):** The two words are dependent. Seeing w₁ significantly changes the probability of seeing w₂ immediately after.

The "Ratio" part comes from dividing the likelihood of H₂ by the likelihood of H₁. We take the Log of this ratio to make the numbers easier to work with (turning multiplication into addition).

### 2. The 2x2 Contingency Table

Like the Chi-square test, LLR uses the four counts of occurrences and non-occurrences:

|  | w₂ present | w₂ absent |
|---|-----------|----------|
| **w₁ present** | k₁₁ (Bigram count) | k₁₂ (w₁ without w₂) |
| **w₁ absent** | k₂₁ (w₂ without w₁) | k₂₂ (Neither present) |

### 3. The "Two Worlds" Comparison

LLR imagines two different versions of reality (hypotheses) and asks which one fits your corpus better:

**The Independence World (H₁):** Word w₁ and word w₂ have no relationship. The probability of seeing w₂ is a constant, regardless of whether w₁ just appeared.

**The Dependence World (H₂):** Word w₁ and word w₂ are linked. The probability of seeing w₂ changes significantly depending on whether w₁ is present.

### 4. Why LLR is the "Gold Standard"

The main principle that makes LLR superior to the t-test or Chi-square for language is its sensitivity to rare events.

The t-test assumes that word frequencies follow a "Normal Distribution" (the bell curve). However, language is "Zipfian"—most words are extremely rare. The t-test often fails to handle these rare words correctly.

**LLR is a "likelihood" test based on binomial distributions.** It doesn't care if the data is rare or frequent; it only cares about the evidence of a shift in probability. This is why it works perfectly for both high-frequency pairs like "of the" and rare technical terms that only appear 5 times.

### 5. Summary Principle

The LLR doesn't just ask "Do these words appear together often?" It asks: **"Does the presence of w₁ significantly change our expectation of seeing w₂?"** If the answer is a statistically "loud" yes, you have found a collocation.

---

## Topic 16: LLR Mathematical Details

**You:** Yes, can you explain the calculations for LLR more?

**Gemini:** To calculate the Log-Likelihood Ratio (LLR), you need to compare two mathematical models (hypotheses) using the raw counts from your corpus.

In LLR, we assume that the distribution of words follows a **Binomial Distribution**. We calculate how well the "Independence Model" fits our data versus how well the "Dependence Model" fits.

### 1. Mapping Your Counts to the LLR Table

For a specific bigram (w₁, w₂), you need these four values:

- **k₁** (Successes for w₁): Count of (w₁, w₂)
- **n₁** (Total trials for w₁): Total count of w₁
- **k₂** (Successes when NOT w₁): Count of w₂ appearing without w₁ before it
- **n₂** (Total trials when NOT w₁): Total tokens in corpus minus total count of w₁

### 2. The LLR Logic (The "Two Worlds" Comparison)

The LLR measures the ratio between two hypotheses:

**Hypothesis 1 (Independence):** The probability of seeing w₂ is the same regardless of whether w₁ preceded it. (p₁ = p₂ = p)

**Hypothesis 2 (Dependence):** The probability of seeing w₂ is different when w₁ precedes it. (p₁ ≠ p₂)

### 3. The Calculation Steps

You calculate the "Likelihood" (L) using the binomial distribution formula. In its log form, the function H(k,n) is often used:

$$H(k,n) = k \log\left(\frac{k}{n}\right) + (n-k)\log\left(1 - \frac{k}{n}\right)$$

The final LLR score is calculated as:

$$LLR = 2[H(k_1, n_1) + H(k_2, n_2) - H(k_1 + k_2, n_1 + n_2)]$$

### 4. Why LLR is Superior for Your "Long Tail"

Remember how 74% of your bigrams appeared less than 5 times?

The t-test assumes a normal distribution, which "breaks" when counts are very small. It might give a high score to a rare pair just because they appeared twice.

**LLR is a "likelihood" test.** It asks if the distribution of w₂ is meaningfully different after w₁. If the sample size is tiny, the LLR score will naturally stay low because there isn't enough evidence to claim a strong dependency.

### 5. How to Interpret the LLR

If you add an LLR column to your dataframe:

- **Values > 10.83:** Extremely significant (p < 0.001). These are your primary collocations
- **Values around 3.84:** Marginally significant
- **Comparing to T-test:** You will likely find that the "Top 20" for both tests are very similar for high-frequency words, but for mid-frequency words, the LLR will provide a much more "realistic" ranking than the t-test

Since you've already filtered your data to N≥5, the LLR and t-test should both perform well, but the **LLR will be more mathematically "honest"** about the pairs that are closer to that threshold.

---

## End of Conversation

This conversation provides comprehensive coverage of NLP fundamentals for your assignment, including token analysis, TF-IDF, collocations, and statistical methods for discovering word relationships.

**Key Takeaways:**
1. Choose appropriate visualization methods based on your corpus characteristics
2. Understand TF-IDF and its components for meaningful feature extraction
3. Apply appropriate statistical tests (t-test, LLR) for collocation discovery
4. Use proper data structures (nested dictionaries) for efficient computation
5. Consider the long tail of rare events in your data
6. Justify your threshold choices for academic integrity
