library(readr)
library(tidyverse)
setwd("~/R Data/DSSTox/DSSTox_CCD_dump_12092025")
DSSToxCCDdump <- read_csv("R Data/DSSTox/DSSTox_CCD_dump_12092025/DSSToxCCDdump.csv")

index = 1
count = 1
end = ceiling(nrow(DSSToxCCDdump) / 100000) 
while(index < end+1) {
  begin = index
  assign(paste0("DSSToxCCDdump",index),DSSToxCCDdump[count:(count+99999),])
  if ((count+99999) > nrow(DSSToxCCDdump)) assign(paste0("DSSToxCCDdump",index),DSSToxCCDdump[count:nrow(DSSToxCCDdump),])
  index = index + 1
  count = count + 100000
}

for (i in 1:end) {
  write_csv(get(paste0("DSSToxCCDdump",i)),paste0("DSSToxCCDdump",i,".csv"))
}




