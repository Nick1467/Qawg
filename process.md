# generator experiement class
class exp(ExperimentProgram):
    def init
    def body

# load config
program = exp(
    awg_fc_cfg,
    name="awg_fc_sequence_sweep",
    final_delay_s=FINAL_DELAY,
)
program.REMOVE_DC_OFFSET = True
compiled = program.compile(hardware=experiment)
delay_result = program.acquire(
    n_average=average,
    filter_type="boxcar",
)

delay_result should be iqdata

# get sweep parameter
fc = self.add_sweep(
    "fc",
    LinearSweep(
        cfg["fc_start"],
        cfg["fc_stop"],
        cfg["fc_points"],
    ),
)
