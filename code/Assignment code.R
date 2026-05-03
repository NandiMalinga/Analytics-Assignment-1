# Analytics assignment
data <- read.csv("listings.csv")
des <- read.csv("variable_descriptions.csv")

host_listings_count <- data$host_listings_count
dev.off()
# distribution of host_listing_count
hist(host_listings_count, breaks = 400, ylim = c(0, 10000))
abline(v = median(host_listings_count), col = "red")
abline(v = mean(host_listings_count, col = "turquoise4"))
abline(v = 9, col = "hotpink")


mean(host_listings_count)
min(host_listings_count)
max(host_listings_count)


