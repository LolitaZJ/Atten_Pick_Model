Usage

Date: 2022.09.06

Ji Zhang

# Atten_PK_Model 
Atten_PK_Model is an attention-based neural network to achieve the phase picking. 
Earthquake detection and phase picking using machine learning with an attention mechanism.


## 1. Set envirnoment 
> tensorflow >= 2.0  
> python >= 3.8  
> 
## 2. Network Architecture
![Network Architecture](/Network_Architecture.png)

## 3. build_pk_model
- Based on Conv1D layers     
- Input size (N,C)  
- Output size (N,1)

**Notes:** N must be an integer tims of 8.

## 4. pk_model
- Based on Conv2D layers     
- Input size (N,1,C)  
- Output size (N,1,1)   
> You can set filters, kernel_size, and depths. Default parameters: `nb_filter`=8, `kernel_size`=(7,1),`depths`=5.

**Notes:** N can be any sizes.

## 5. Fast run!
### a) build_pk_model
> train   
`python Attention_PK_MASTER.py --model_name=bulid_pk_model`

### b) pk_model
> train   
`python Attention_PK_MASTER.py --model_name=pk_model`

**Notes:**   
- Here, we just emphasize the building of the model, and the data is the random number generated.   
- The real training data should be seismic data, and the label is the Gaussian distribution of its corresponding seismic phase, similar to <code>[PhaseNet](https:https://github.com/wayneweiqiang/PhaseNet)</code> or use `https://github.com/wayneweiqiang/PhaseNet`. 
- You can load <code>[STEAD Data](https://github.com/smousavi05/STEAD)</code> or use `https://github.com/smousavi05/STEAD` to get seismic dataset for training. 

## Related papers:
- Zhu, W., & Beroza, G. C. (2019). PhaseNet: a deep-neural-network-based seismic arrival-time picking method. Geophysical Journal International, 216(1), 261-273.
- Mousavi, S. M., Ellsworth, W. L., Zhu, W., Chuang, L. Y., & Beroza, G. C. (2020). Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking. Nature communications, 11(1), 1-12.
