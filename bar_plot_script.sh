for i in $(seq "1000")
do
   echo "Layer 1 Iteration $i"
   python3 top_module.py --arch ./inputs_barchart/architecture_eyeriss_tl.yaml --map ./inputs_barchart/mapping_eyeriss_tl_1.yaml --set ./inputs_barchart/settings_1.yaml --mempool ./inputs_barchart/memory_pool_eyeriss_tl.yaml
done

for i in $(seq "1000")
do
   echo "Layer 2 Iteration $i"
   python3 top_module.py --arch ./inputs_barchart/architecture_eyeriss_tl.yaml --map ./inputs_barchart/mapping_eyeriss_tl_2.yaml --set ./inputs_barchart/settings_2.yaml --mempool ./inputs_barchart/memory_pool_eyeriss_tl.yaml
done

for i in $(seq "1000")
do
   echo "Layer 9 Iteration $i"
   python3 top_module.py --arch ./inputs_barchart/architecture_eyeriss_tl.yaml --map ./inputs_barchart/mapping_eyeriss_tl_9.yaml --set ./inputs_barchart/settings_9.yaml --mempool ./inputs_barchart/memory_pool_eyeriss_tl.yaml
done

for i in $(seq "1000")
do
   echo "Layer 10 Iteration $i"
   python3 top_module.py --arch ./inputs_barchart/architecture_eyeriss_tl.yaml --map ./inputs_barchart/mapping_eyeriss_tl_10.yaml --set ./inputs_barchart/settings_10.yaml --mempool ./inputs_barchart/memory_pool_eyeriss_tl.yaml
done

for i in $(seq "1000")
do
   echo "Layer 18 Iteration $i"
   python3 top_module.py --arch ./inputs_barchart/architecture_eyeriss_tl.yaml --map ./inputs_barchart/mapping_eyeriss_tl_18.yaml --set ./inputs_barchart/settings_18.yaml --mempool ./inputs_barchart/memory_pool_eyeriss_tl.yaml
done

for i in $(seq "1000")
do
   echo "Layer 19 Iteration $i"
   python3 top_module.py --arch ./inputs_barchart/architecture_eyeriss_tl.yaml --map ./inputs_barchart/mapping_eyeriss_tl_19.yaml --set ./inputs_barchart/settings_19.yaml --mempool ./inputs_barchart/memory_pool_eyeriss_tl.yaml
done

for i in $(seq "1000")
do
   echo "Layer 31 Iteration $i"
   python3 top_module.py --arch ./inputs_barchart/architecture_eyeriss_tl.yaml --map ./inputs_barchart/mapping_eyeriss_tl_31.yaml --set ./inputs_barchart/settings_31.yaml --mempool ./inputs_barchart/memory_pool_eyeriss_tl.yaml
done

for i in $(seq "1000")
do
   echo "Layer 32 Iteration $i"
   python3 top_module.py --arch ./inputs_barchart/architecture_eyeriss_tl.yaml --map ./inputs_barchart/mapping_eyeriss_tl_32.yaml --set ./inputs_barchart/settings_32.yaml --mempool ./inputs_barchart/memory_pool_eyeriss_tl.yaml
done
