The data in sample.npy has a dimensions of (1,16,14,160,256)~(B,T,C,H,W)

By using ``python inference.py'', the T+1 to T+4 forecast will be generated.
By using forecasted satellite images from T+1 to T+4, it is possible to generate forecasts for T+5 to T+8.