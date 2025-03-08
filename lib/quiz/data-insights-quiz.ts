import type { QuizQuestion } from "../types"
export const beginner: QuizQuestion[] = [
  {
    question: "What is the primary goal of data insights analysis?",
    options: [
      { id: "a", text: "To clean messy data" },
      { id: "b", text: "To extract meaningful patterns and knowledge from data" },
      { id: "c", text: "To create fancy visualizations" },
      { id: "d", text: "To build machine learning models" },
    ],
    correctAnswer: "b",
    explanation:
      "The primary goal of data insights analysis is to extract meaningful patterns, trends, and knowledge from data to support decision-making.",
  },
  {
    question: "Which of the following is an example of a descriptive statistic?",
    options: [
      { id: "a", text: "P-value" },
      { id: "b", text: "Confidence interval" },
      { id: "c", text: "Mean" },
      { id: "d", text: "Hypothesis test" },
    ],
    correctAnswer: "c",
    explanation:
      "The mean is a descriptive statistic that summarizes a characteristic of a dataset, in this case the central tendency.",
  },
  {
    question: "What does the standard deviation measure?",
    options: [
      { id: "a", text: "The average value in a dataset" },
      { id: "b", text: "The spread or dispersion of data points" },
      { id: "c", text: "The relationship between two variables" },
      { id: "d", text: "The most frequent value in a dataset" },
    ],
    correctAnswer: "b",
    explanation:
      "Standard deviation measures the amount of variation or dispersion of a set of values from the mean. A low standard deviation indicates that values tend to be close to the mean, while a high standard deviation indicates values are spread out over a wider range.",
  },
  {
    question: "In a dataset, what is an outlier?",
    options: [
      { id: "a", text: "A data point near the median" },
      { id: "b", text: "A data point that differs significantly from other observations" },
      { id: "c", text: "The most common value" },
      { id: "d", text: "A missing value" },
    ],
    correctAnswer: "b",
    explanation:
      "An outlier is a data point that differs significantly from other observations in the dataset. Outliers may be due to variability in the measurement or may indicate experimental errors.",
  },
  {
    question: "What is a frequency distribution?",
    options: [
      { id: "a", text: "A list of all possible values in a dataset" },
      { id: "b", text: "A way to organize and summarize data by showing how often each value occurs" },
      { id: "c", text: "The relationship between two variables" },
      { id: "d", text: "A statistical test for comparing datasets" },
    ],
    correctAnswer: "b",
    explanation:
      "A frequency distribution organizes and summarizes data by showing the number of times each value occurs in the dataset, helping to identify patterns and trends.",
  },
]

export const intermediate: QuizQuestion[] = [
  {
    question: "What is the difference between a parameter and a statistic?",
    options: [
      { id: "a", text: "A parameter is a sample value; a statistic is a population value" },
      { id: "b", text: "A parameter is a population value; a statistic is a sample value" },
      { id: "c", text: "A parameter is calculated; a statistic is estimated" },
      { id: "d", text: "There is no difference; the terms are interchangeable" },
    ],
    correctAnswer: "b",
    explanation:
      "A parameter is a numerical characteristic of a population (e.g., population mean μ), while a statistic is a numerical characteristic calculated from a sample (e.g., sample mean x̄).",
  },
  {
    question: "What is the central limit theorem?",
    options: [
      { id: "a", text: "The statement that all data tends toward a normal distribution" },
      { id: "b", text: "The statement that the sampling distribution of the mean approaches a normal distribution as sample size increases" },
      { id: "c", text: "The principle that the median is always the best measure of central tendency" },
      { id: "d", text: "The law that states all statistical tests must be centered on the null hypothesis" },
    ],
    correctAnswer: "b",
    explanation:
      "The central limit theorem states that the sampling distribution of the mean approaches a normal distribution as the sample size increases, regardless of the shape of the population distribution.",
  },
  {
    question: "What is the purpose of a confidence interval?",
    options: [
      { id: "a", text: "To provide a range of values where the sample mean falls" },
      { id: "b", text: "To test whether a sample mean differs from a hypothesized value" },
      { id: "c", text: "To provide a range of values likely to contain the population parameter" },
      { id: "d", text: "To determine if a sample is from a normal distribution" },
    ],
    correctAnswer: "c",
    explanation:
      "A confidence interval provides a range of values that is likely to contain the unknown population parameter. For example, a 95% confidence interval means that if we were to take many samples and compute a confidence interval for each sample, about 95% of those intervals would contain the true population parameter.",
  },
  {
    question: "What is a Type I error in hypothesis testing?",
    options: [
      { id: "a", text: "Failing to reject a false null hypothesis" },
      { id: "b", text: "Rejecting a true null hypothesis" },
      { id: "c", text: "Making a computational error in the analysis" },
      { id: "d", text: "Drawing a sample that is not representative" },
    ],
    correctAnswer: "b",
    explanation:
      "A Type I error occurs when we reject a null hypothesis that is actually true. This is also called a 'false positive'.",
  },
  {
    question: "What does a p-value of 0.03 in a hypothesis test tell you?",
    options: [
      { id: "a", text: "The probability that the null hypothesis is true is 0.03" },
      { id: "b", text: "If the null hypothesis is true, the probability of observing a test statistic as extreme as the one observed is 0.03" },
      { id: "c", text: "The probability that the alternative hypothesis is true is 0.97" },
      { id: "d", text: "The test results are 97% accurate" },
    ],
    correctAnswer: "b",
    explanation:
      "A p-value represents the probability of observing a test statistic as extreme as, or more extreme than, the one observed, if the null hypothesis is true. A p-value of 0.03 means that, assuming the null hypothesis is true, there is a 3% chance of observing such extreme results.",
  },
]

export const advanced: QuizQuestion[] = [
  {
    question: "In multiple testing scenarios, why is a correction like Bonferroni necessary?",
    options: [
      { id: "a", text: "To make the tests more powerful" },
      { id: "b", text: "To account for the fact that sample sizes may vary" },
      { id: "c", text: "To control the overall Type I error rate (familywise error rate)" },
      { id: "d", text: "To ensure the tests are appropriate for non-normal data" },
    ],
    correctAnswer: "c",
    explanation:
      "When conducting multiple hypothesis tests, the chance of making at least one Type I error (false positive) increases. Corrections like Bonferroni adjust the significance level to control the overall Type I error rate across all tests.",
  },
  {
    question: "What is the difference between a parametric and a non-parametric statistical test?",
    options: [
      { id: "a", text: "Parametric tests use means; non-parametric tests use medians" },
      { id: "b", text: "Parametric tests make assumptions about population parameters; non-parametric tests make fewer assumptions" },
      { id: "c", text: "Parametric tests are for large samples; non-parametric tests are for small samples" },
      { id: "d", text: "Parametric tests are always more powerful than non-parametric tests" },
    ],
    correctAnswer: "b",
    explanation:
      "Parametric tests make assumptions about the underlying population distribution (often assuming normality), while non-parametric tests make fewer assumptions about the population and are often used when the assumptions of parametric tests are violated.",
  },
  {
    question: "What does a high R² value in a regression model indicate?",
    options: [
      { id: "a", text: "The model is statistically significant" },
      { id: "b", text: "The model has a high proportion of explained variance" },
      { id: "c", text: "The model has many predictors" },
      { id: "d", text: "The model has no multicollinearity" },
    ],
    correctAnswer: "b",
    explanation:
      "A high R² (coefficient of determination) indicates that a large proportion of the variance in the dependent variable is explained by the independent variables in the model. It ranges from 0 to 1, with values closer to 1 indicating better fit.",
  },
  {
    question: "What is the purpose of bootstrapping in statistical analysis?",
    options: [
      { id: "a", text: "To increase the sample size" },
      { id: "b", text: "To estimate the sampling distribution of a statistic through resampling" },
      { id: "c", text: "To remove outliers from the analysis" },
      { id: "d", text: "To transform non-normal data into normal data" },
    ],
    correctAnswer: "b",
    explanation:
      "Bootstrapping is a resampling technique that involves randomly sampling with replacement from the original dataset to estimate the sampling distribution of a statistic, allowing for the calculation of confidence intervals and hypothesis testing without making strong distributional assumptions.",
  },
  {
    question: "What is the curse of dimensionality in data analysis?",
    options: [
      { id: "a", text: "The difficulty in visualizing data with more than three dimensions" },
      { id: "b", text: "The problem that as the number of features increases, the amount of data needed increases exponentially" },
      { id: "c", text: "The issue that high-dimensional data is always multicollinear" },
      { id: "d", text: "The limitation that statistical software can only handle a certain number of variables" },
    ],
    correctAnswer: "b",
    explanation:
      "The curse of dimensionality refers to the phenomena that occur when analyzing data in high-dimensional spaces that do not occur in low-dimensional settings. As the number of features or dimensions grows, the volume of the space increases so quickly that the available data becomes sparse, making statistical analysis more difficult and requiring exponentially more data to maintain the same level of statistical confidence.",
  },
]
