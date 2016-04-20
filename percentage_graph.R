library(dplyr)
library(ggplot2)
library(scales)
library(RColorBrewer)

results=read.csv('class_percentages.csv')

#Set variable levels correctly in df so that ggplot orders things correctly. 
results$catagory = factor(results$catagory, levels=unique(results$catagory))

barplot=ggplot(results, aes(x=type, y=pct*100, fill=catagory))+
  geom_bar(stat='identity')  +
  facet_wrap(~year, nrow=1) +
  theme_bw(base_size = 30) +
  xlab('') + ylab('Percentage of landscape') +
  scale_fill_brewer(palette='Spectral', name='Dead Trees / Pixel') 

jpeg('training_area_bar_plot.jpg', width=1400, height=1000, units='px')
print(barplot)
dev.off()
