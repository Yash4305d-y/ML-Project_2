<h1>❤️ Heart Disease Prediction Engine</h1>

<p>An end-to-end machine learning pipeline built to predict heart disease risk using custom, domain-specific feature engineering.</p>

<blockquote>
  <strong>🚀 Work in Progress:</strong> This project is currently ongoing. The current architecture establishes a baseline using linear methods, and active development is focused on migrating to classification frameworks to scale up model accuracy.
</blockquote>

<hr />

<h2>📊 Preprocessing Pipeline</h2>
<p>To preserve mathematical integrity and avoid data leakage, data flows through a strict order of operations:</p>

<ul>
  <li><strong>1. Raw Ratios:</strong> Calculates <code>HR_BP_Ratio</code> (<code>MaxHR / RestingBP</code>) on raw numbers first to avoid breaking division math after scaling.</li>
  <li><strong>2. One-Hot Encoding:</strong> Converts categorical columns into clean binary vectors using <code>drop_first=True</code> to protect the linear model from multicollinearity.</li>
  <li><strong>3. Standardization:</strong> Applies <code>StandardScaler</code> to all continuous columns so they share a uniform scale centered around zero.</li>
  <li><strong>4. Interaction Terms:</strong> Generates advanced columns (<code>MaxHR * Oldpeak</code>, <code>Age * Cholesterol</code>, and <code>Oldpeak²</code>) <em>after</em> scaling to cleanly map non-linear clinical risks.</li>
</ul>

<hr />

<h2>💡 Core Architecture Concept</h2>
<p>The core concept centers around equipping a standard, rigid model with advanced visual capabilities. By generating interaction matrices and polynomial boundaries post-standardization, the model is forced to recognize how vital features compound together—like high heart rates paired with abnormal electrical readouts—allowing a simple architecture to capture multi-dimensional clinical risks without destabilizing the feature weight space.</p>

<hr />

<h2>📈 Performance Baseline</h2>

<table>
  <tr>
    <th>Metric</th>
    <th>Current Baseline Value</th>
    <th>Target Benchmark</th>
  </tr>
  <tr>
    <td><strong>Linear Model R² Score</strong></td>
    <td><code>0.5304</code></td>
    <td><em>Baseline Set</em></td>
  </tr>
</table>

<p><strong>Analysis:</strong> The engineered interaction terms successfully injected non-linear capabilities into the architecture, maximizing performance boundaries for a linear structure tracking a binary target.</p>

<hr />

<h2>🔮 Upcoming Improvements</h2>
<ul>
  <li><strong>Model Migration:</strong> Transitioning the baseline pipeline to classification models (like <code>LogisticRegression</code> or ensemble tree classifiers) to natively capture the binary decision boundaries.</li>
  <li><strong>Regularization:</strong> Introducing Ridge (L2) and Lasso (L1) parameters to penalize and optimize the newly added high-dimensional interaction weights.</li>
</ul>
