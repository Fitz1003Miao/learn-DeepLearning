CLASS_NUM = 2



fcLayers_param_name = ('inputD', 'outputD', 'KeepPro', 'name')
fcLayers_params = [dict(zip(fcLayers_param_name, fcLayers_param)) for fcLayers_param in 
                    [(256 * 6 * 6, 4096, 0.5, 'fc1'), 
                     (4096, 4096, 0.5, 'fc2'), 
                     (4096, CLASS_NUM, None, 'fc3')]]