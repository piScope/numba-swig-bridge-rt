"""MPI communicator usage with the installed nsb-sample11 package."""
from mpi4py import MPI
from numba import njit
import nsb_rt as nsb
import nsb_sample11 as sample
import nsb_sample11_registration  # Registers ordinary adapters and MPI provider.


@njit
def inspect(comm):
    return comm.rank, comm.size


@njit
def through_native_constructor(comm):
    consumer = sample.CommConsumer(comm)
    native_comm = consumer.GetComm()
    return inspect(native_comm)


@njit
def make_consumer(comm):
    # The caller must keep the MPI resource alive during consumer use.
    return nsb.own_this(sample.CommConsumer(comm))


@nsb.director(state={"comm": MPI.Comm})
class RankCallback(sample.CommCallback):
    def __init__(self, comm):
        super().__init__()
        self.comm = comm

    @nsb.override
    def Value(self):
        return inspect(self.comm)[0]


def main():
    comm = MPI.COMM_WORLD
    print("direct:", inspect(comm))
    print("native constructor and helper:", through_native_constructor(comm))
    consumer = make_consumer(comm)
    callback = RankCallback(comm)
    try:
        print("director state rank:", consumer.Apply(callback))
    finally:
        # End native consumer use before detaching its callback state.
        del consumer
        callback.close()


if __name__ == "__main__":
    main()
