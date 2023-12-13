#encoding categoricals


# data --------------------------------------------------------------------
glimpse(fn_luc18)
glimpse(fn_luc15)
glimpse(fn_luc09)

# Identify character or factor variables
fct_luc18 <- sapply(fct_luc18, function(x) as.factor(x))
fct_luc15 <- sapply(fct_luc15, function(x) as.factor(x))
fct_luc09 <- sapply(fct_luc09, function(x) as.factor(x))


### glm can handle categoricals direclty without prior encoding.