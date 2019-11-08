package internals;

import org.apache.spark.Partitioner;

public class EvenPartitioner extends Partitioner {
    private int numPartitions;

    public EvenPartitioner(int i) {
        this.numPartitions = i;
    }

    @Override public int numPartitions() {
        return numPartitions;
    }

    @Override public int getPartition(Object key) {
        if (Integer.valueOf(key.toString()) % 2 == 0)
            return 0;
        else
            return 1;
    }
}
