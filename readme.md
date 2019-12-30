# Code repository for "Decomposition ensemble model based on variational mode decomposition and long short-term memory for streamflow forecasting"

## Jianyi Zuo 
E-mail:zuojianyi@outlook.com 
Github:https://github.com/zjy8006



This study relied heavily on open-source software. Pandas (McKinney, 2010) and numpy (Stéfan et al., 2011) were used to manage and process streamflow data. Matlab was used to perform the streamflow decomposition tasks and compute the PACF of subsignals. The Matlab implementations of [VMD](https://ieeexplore.ieee.org/document/6655981) and [EEMD](https://doi.org/10.1142/S1793536909000047) come from Dragomiretskiy and Zosso (2014) and Wu and Huang (2009), respectively. The [DWT](https://www.mathworks.com/help/wavelet/ref/dwt.html) was performed based on the Matlab build-in toolbox (“Wavelet 1-D” in “Wavelet Analyzer”). The GBRT model in [scikit-optimize](https://scikit-optimize.github.io/) (Pedregosa et al., 2011) was used to measure the importance of the decomposed subsignals. Matplotlib (Hunter, 2007) was used to draw figures, and [TensorFlow](https://tensorflow.google.cn/) (Abadi et al., 2016) was used to train the LSTM models. These open-source software also were partly used by previous researchers, e.g., Kratzert et al. (2018).

## How to validate the research results

1. Clone this repository form [Github](https://github.com/zjy8006/DailyStreamflowForecastAutoReg).
   ```
   git clone https://github.com/zjy8006/DailyStreamflowForecastAutoReg
   ```
2. Open MATLAB for streamflow decomposition. Go to the root directory.
   ```
   cd Local_disk:/DailyStreamflowForecastAutoReg/
   ```
3. Open this repository with [vscode](https://code.visualstudio.com/) for other tasks. Install **code runner** extension and enable "Run in Terminal". Run code with ![Run code](/config/run_code.png)

## Trend and abrupt shift detection

* Run **"/results_analyze/plot_trend_abrupt.py"** for trend and abrupt shift detection.

## Streamflow decomposition

* Run **"/tools/RUN_VMD.m"** for VMD of streamflow.
* Run **"/tools/RUN_EEMD.m"** for EEMD of streamflow.
* Run **"/tools/RUN_DWT.m"** for DWT of streamflow.

## Compute Partial autocorrelation coefficient

* Run **"/tools/compute_pacf.m"**

## Importance measurement

* Run **"/tools/feature_selection.py"**

## Modelling process

### Generate samples

* Run **"/yx_orig/projects/generate_orig_samples.py"**
* Run **"/yx_eemd/projects/generate_eemd_samples.py"**
* Run **"/yx_vmd/projects/generate_vmd_samples.py"**
* Run **"/yx_wd/projects/generate_wd_samples.py"**
* Run **"/zjs_orig/projects/generate_orig_samples.py"**
* Run **"/zjs_eemd/projects/generate_eemd_samples.py"**
* Run **"/zjs_vmd/projects/generate_vmd_samples.py"**
* Run **"/zjs_wd/projects/generate_wd_samples.py"**

### Tune LSTM models

* Run **"/yx_orig/projects/run_orig_lstm.py"**
* Run **"/yx_eemd/projects/run_eemd_lstm.py"**
* Run **"/yx_vmd/projects/run_vmd_lstm.py"**
* Run **"/yx_wd/projects/run_dwt_lstm.py"**
* Run **"/zjs_orig/projects/run_orig_lstm.py"**
* Run **"/zjs_eemd/projects/run_eemd_lstm.py"**
* Run **"/zjs_vmd/projects/run_vmd_lstm.py"**
* Run **"/zjs_wd/projects/run_dwt_lstm.py"**

## Results analysis

* Figure 6: run **"/results_analyze/plot_trend_abrupt.py"**
* Figure 7: run **"/results_analyze/plot_aliasing.py"**
* Figure 8: run **"/results_analyze/plot_feature_selection.py"**
* Figure 9: run **"/results_analyze/plot_pacfs.py"**
* Figure 10: run **"/results_analyze/plot_learn_rate.py"**
* Figure 11: run **"/results_analyze/plot_model_structure.py"**
* Figure 12, 13 and 14: run **"/results_analyze/plot_training_development_metrics.py"**
* Figure 15: run **"/results_analyze/plot_forecsing_testing_metrics.py"**
* Figure 16 and 17: run **"/results_analyze/plot_hind_forecast_scatters.py"**
* Figure 18: run **"/results_analyze/Pearson_corr_subsignals.py"**
* Figure 19: run **"/results_analyze/plot_subsignals_frequency.py"**
* Figure 20, 21 and 22: run **"/results_analyze/plot_boundary_effect_vmd_eemd_dwt.py"**

## Cite us

* Please cite the authors of open-source software, such as EEMD, VMD, Pandas, Matplotlib, Numpy, Scikit-learn, TensorFlow, .etc, if you used them.
* Please cite us if you use this repository for further research.
```
Cite all versions? You can cite all versions by using the DOI 10.5281/zenodo.3595150 (Add to Citavi project by DOI). This DOI represents all versions, and will always resolve to the latest one.
```



## Reference

* Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G.S., Davis, A., Dean, J., Devin, M., Ghemawat, S., Goodfellow, I., Harp, A., Irving, G., Isard, M., Jia, Y., Jozefowicz, R., Kaiser, L., Kudlur, M., Levenberg, J., Mane, D., Monga, R., Moore, S., Murray, D., Olah, C., Schuster, M., Shlens, J., Steiner, B., Sutskever, I., Talwar, K., Tucker, P., Vanhoucke, V., Vasudevan, V., Viegas, F., Vinyals, O., Warden, P., Wattenberg, M., Wicke, M., Yu, Y., Zheng, X., 2016. TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems.
* Dragomiretskiy, K., Zosso, D., 2014. Variational Mode Decomposition. IEEE Trans. Signal Process. 62 (3), 531–544.
* Hunter, J.D., 2007. Matplotlib. A 2D Graphics Environment. Computing in Science & Engineering 9, 90–95.
* Kratzert, F., Klotz, D., Brenner, C., Schulz, K., Herrnegger, M., 2018. Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks. Hydrol. Earth Syst. Sci. 22 (11), 6005–6022.
* McKinney, W., 2010. Data Structures for Statistical Computing in Python, pp. 51–56.
* Stéfan, v.d.W., Colbert, S.C., Varoquaux, G., 2011. The NumPy Array: A Structure for Efficient Numerical Computation. A Structure for Efficient Numerical Computation. Comput. Sci. Eng. 13 (2), 22–30.
* Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., Duchesnay, É., 2011. Scikit-learn. Machine Learning in Python. Journal of Machine Learning Research 12, 2825–2830.
* Tim, H., MechCoder, Gilles, L., Iaroslav, S., fcharras, Zé Vinícius, cmmalone, Christopher, S., nel215, Nuno, C., Todd, Y., Stefano, C., Thomas, F., rene-rex, Kejia, (K.) S., Justus, S., carlosdanielcsantos, Hvass-Labs, Mikhail, P., SoManyUsernamesTaken, Fred, C., Loïc, E., Lilian, B., Mehdi, C., Karlson, P., Fabian, L., Christophe, C., Anna, G., Andreas, M., and Alexander, F.: Scikit-Optimize/Scikit-Optimize: V0.5.2, Zenodo, 2018.
* Wu, Z., Huang, N.E., 2009. Ensemble Empirical Mode Decomposition: a Noise-Assisted Data Analysis Method. Adv. Adapt. Data Anal. 01 (01), 1–41.

