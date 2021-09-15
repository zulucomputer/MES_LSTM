for country in 'Angola' 'Botswana' 'Comoros' 'Democratic_Republic_of_Congo' 'Eswatini' 'Lesotho' 'Madagascar' 'Malawi' 'Mauritius' 'Mozambique' 'Namibia' 'South_Africa' 'Tanzania' 'Zambia' 'Zimbabwe'
    do
    jbsub -cores 1x4+0 -out logs/log_${country}.txt -mem 10G -queue x86_6h \
    "source activate rnn0; \
    python /dccstor/eevdata/mathonsi/tutorials/MES_LSTM/run_multi_MES_LSTM.py \
    --country ${country}"
done

# source activate rnn0
# python /dccstor/eevdata/mathonsi/tutorials/MES_LSTM/run_multi_MES_LSTM.py  --country ${country} --thresh ${thresh}