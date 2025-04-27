------------------------------------------------------------------------------
Bayesian Reasoning and Machine Learning, David Barber
## Bayes' Rule
$$
p(x|y) = \frac{p(y|x) \, p(x)}{p(y)}
$$

## Independence of Variables \(x\) and \(y\)
Variables are independent when knowing about one does not provide any additional information about the other.

$$
p(x, y) = p(x) \, p(y)
$$

=> Spectrum 1 is independent of Spectrum 2 since knowing about Spectrum 1 doesn't provide extra information about Spectrum 2, and vice versa.

## Conditional Independence of Two Sets of Variables \(X\) and \(Y\) Given \(Z\)
\(X\) and \(Y\) are **conditionally independent** given all states of \(Z\).

$$
p(X, Y | Z) = p(X | Z) \, p(Y | Z)
$$

=> A set of observations \(X\) is independent of a set of observations \(Y\) given \(Z\).

## Generative Model \(p(D|\theta)\)
- **Prior**: $p(\theta_i)$ (for \(i=1,2,3,4\))
- **Likelihood**: $p(D|\theta_i)$
- **Posterior**: $p(\theta|D) = \frac{p(D|\theta_i) \, p(\theta_i)}{p(D)}$

where $p(D) = \int_{\theta} p(D|\theta_i) \, p(\theta_i)$

Goal:  Given a set of observations $D$, find $i$ such that $\arg\max_i \, p(\theta_i | D)$

## Belief Network
    (θi) --------> x1
        \-------> x2 
                  .
                  .
                  .
        \-------> xn



- There are arrows connecting $\theta$ to each $x_i$.
- x is dependent on $\theta$ but independent of $x_j$ for $i \neq j$.
