# Functions

# This function will perform the fft of the data given, perform abs, arrange per
# in descending order (max first), and return the pos requested in Hz
# INPUTS:
# data
# pos = position starting at 0 i.e 0,1,2...
# fs= frequency of sampling in Hz. Usually 1/(t_sampling).
# OUTPUTS:


fft_top3 <-function (z, fs){
  window <- length(z)
  power <- data.frame(freq=fs/window*(0:(window-1)),amp=1/window*abs(fft(z)))
  power <- slice(power,1:(window/2))
  power <- arrange(power,desc(amp))
  # Return only the mean and 3 most significant amplitudes
  return (slice(power,1:4))
}