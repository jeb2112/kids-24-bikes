# todo
# sheet cells are all hard-coded, have to be updated manually whnever sheet is changed!

# packages
library(googledrive)
library(plotrix)
library(googlesheets)
library(ggplot2)
library(magrittr)
library(dplyr)
library(tidyr)
library(broom)
library(nlstools)
library(lme4)
library(reshape2)
library(cluster)
library(gridExtra)
library(gridGraphics)
library(factoextra)
PLOT.PRINT=TRUE

# calculate com
calc_com <- function(coords,anthro) {
  com_leg = c((coords$BB[1]-coords$saddle[1])/2+coords$saddle[1],(coords$saddle[2]-coords$BB[2])/2+coords$BB[2])
  com_torso = c((coords$shldr[1]-coords$saddle[1]/2)+coords$saddle[1],(coords$shldr[2]-coords$saddle[2])/2+coords$saddle[2])
  com_arm = c((coords$stem[1]-coords$shldr[1])/2+coords$shldr[1],(coords$shldr[2]-coords$stem[2])/2+coords$stem[2])
  com <- c((com_leg[1]*anthro$leg + com_torso[1]*anthro$torso + com_arm[1]*anthro$arm)/sum(anthro),
           (com_leg[2]*anthro$leg + com_torso[2]*anthro$torso + com_arm[2]*anthro$arm)/sum(anthro))
  return(com)
  
}

# save plot to file and upload to drive
plotprint <- function(plot.obj,filename,width,height,todrive=FALSE) {
  # check for existing png device
  if ('png' %in% names(x<-dev.list())) {
    dev.set(dev.list()['png'])
    # but, no idea how to add the filename to the existing png device, so turn if off and set a new one
    dev.off(dev.list()['png'])
  } 
  # create new png device
  png(file = sprintf('/home/jbishop/src/R/%s.png',filename),width=width,height=height)
  if (is.ggplot(plot.obj)) {
    plot(plot.obj)
  } else if (is.grob(plot.obj)) {
    grid.arrange(plot.obj)
  }
  # reset graph device back to rstudio
  dev.off(dev.list()['png'])
  dev.set(dev.list()['RStudioGD'])
  # also upload plot to googledrive. another bizrre problem. if the file exists, need update to overwrite, but if it 
  # doesn't exist, update won't work need upload. 
  if (todrive) {
    drive.get <- drive_get(sprintf('%s.png',filename))
    # another problem. if the file doesn't exist, the drive_get can't return a name to test. just test for length
    # if (drive.get$name == 'weight-price.png') { 
    if (length(drive.get$name > 0)) {
      drive_update(sprintf('%s.png',filename),sprintf('/home/jbishop/src/R/%s.png',filename))
    } else {
      drive_upload(sprintf('/home/jbishop/src/R/%s.png',filename),sprintf('%s.png',filename))
    }
  }
}

# plot the bike and com
plot_bike <- function(bike_names,coords_xy,get.grob=FALSE) {
  plot.title = c(bike_names)
  plot.new() 
  for (name in bike_names) {
    xy = coords_xy[[name]]
    # create blank plot
    if (name==bike_names[1]) {
      col=1
      plot(c(0,xy$BB[1]),c(0,xy$BB[2]),type="n",asp=1,xlim=c(0,100),ylim=c(-10,70),lwd=2,xlab="x (cm)",ylab="y (cm)")
      legend("topleft",bike_names,text.col=1:length(bike_names),cex=1)
    } else {
      col <- col+1
    }
    # chainstay
    lines(c(0,xy$BB[1]),c(0,xy$BB[2]),col=col,lwd=2)
    # seattube
    if (!is.na(xy$saddle[1])) {
      lines(c(xy$BB[1],xy$saddle[1]),c(xy$BB[2],xy$saddle[2]),col=col,lwd=2)
    } else {
      lines(c(xy$BB[1],xy$stXtt[1]),c(xy$BB[2],xy$stXtt[2]),col=col,lwd=2)
    }
    # downtube
    lines(c(xy$BB[1],xy$htXdt[1]),c(xy$BB[2],xy$htXdt[2]),col=col,lwd=2)
    # toptube
    lines(c(xy$stXtt[1],xy$htXtt[1]),c(xy$stXtt[2],xy$htXtt[2]),col=col,lwd=2)
    #headtube
    lines(c(xy$htXtt[1],xy$htXdt[1]),c(xy$htXtt[2],xy$htXdt[2]),col=col,lwd=2)
    #seatstay
    lines(c(0,xy$stXtt[1]),c(0,xy$stXtt[2]),col=col,lwd=2)
    # fork
    lines(c(xy$htXdt[1],xy$axle[1]),c(xy$htXdt[2],xy$axle[2]),col=col,lwd=2)
    # stem
    lines(c(xy$htXtt[1],xy$stem[1]),c(xy$htXtt[2],xy$stem[2]),col=col,lwd=2)
    # seat
    if (!is.na(xy$saddle[1])) {
      lines(c(xy$saddle[1]-3,xy$saddle[1]+3),c(xy$saddle[2],xy$saddle[2]),col=col,lwd=2)
    }
    # com
    if (!is.na(xy$com[1])) {
      dcom <- draw.circle(xy$com[1],xy$com[2],2,col=col)
    }
    # cob
    lines(c(xy$axle[1]/2,xy$axle[1]/2),c(0,70),col=col,lwd=1,lty=2)
    # dc <-draw.circle(0,0,14*2.54,lty=1,lwd=1,border=col)
    # dc2 <- draw.circle(xy$axle[1],0,14*2.54,lty=1,lwd=1,border=col)
  }
  # for some reason this is creating extra plots in the plot pane, one for each nname in bike_names, 
  # even though this is outside the bike_names loop, and 1 of them is blank and bike_names - 1 are duplicate
  if (get.grob) {
    bike.grob <- grab_grob()
  } else {
    bike.grob <- NULL
  }
}


# coordinates of bicycle. origin is rear axle
# no useful way to use two coords as a single list element in the frame, because that becomes a list in a list
# and referencing individual values of a list in a list is a cumbersome unlist (frameelement )[[1]]
bike_coords = function(geo,anthro) {
  coords <- data.frame(BB=c(geo$rear.centre,-geo$bob.drop),
                       axle=c(geo$wheelbase,0))
  if (!is.na(geo$reach) & !is.na(geo$stack)) {
    coords$htXtt <- c(coords$BB[1]+geo$reach,coords$BB[2]+geo$stack)
  } else {
    print("Missing reach/stack") # generally can't compute missing reach/stack without good fork info
  }
  coords$htXdt <- c(coords$htXtt[1]+cos(geo$HA)*geo$head.tube,coords$htXtt[2]-sin(geo$HA)*geo$head.tube)
  
  # top tube needed for drawing purpose only, will be approximated by stand-over height, which is assumed to be
  # at the bottom bracket. need to rationalize these numbers from the photos
  fit_tt <- lm(c(coords$htXtt[2],geo$standover) ~ c(coords$htXtt[1],coords$BB[1]))
  slope_st = -tan(geo$SA)
  intcpt_st = -slope_st * coords$BB[1]
  x_stXtt = (intcpt_st - fit_tt$coefficients[1]) / (fit_tt$coefficients[2] - slope_st)
  y_stXtt = slope_st * x_stXtt + intcpt_st
  
  coords$stXtt <- c(x_stXtt,y_stXtt)
  coords$stem <- c(coords$htXtt[1]+geo$stem/10 * cos(geo$stemA + pi/2-geo$HA),
                   coords$htXtt[2]+geo$stem/10 * sin(geo$stemA + pi/2-geo$HA))
  coords$saddle <- c(coords$BB[1]+cos(geo$SA)*geo$crank.length-cos(geo$SA)*anthro$leg*0.883,
                     coords$BB[2]-sin(geo$SA)*geo$crank.length + sin(geo$SA)*anthro$leg*0.883)
  saddle2bar <- sqrt((coords$stem[1]-coords$saddle[1])^2 + (coords$stem[2]-coords$saddle[2])^2)
  # given seat and bar, find the angle of the torso. add torso curvature option
  A_torso = acos((saddle2bar^2+anthro$torso^2-anthro$arm^2)/(2*saddle2bar*anthro$torso))
  # shoulder at the apex of torso arm triangle
  coords$shldr <- c(coords$saddle[1]+anthro$torso*cos(A_torso),coords$saddle[2]+anthro$torso*sin(A_torso)) 
  row.names(coords) <- c('x','y')
  return(coords)
}

read.sheet = function(ss,ws,main.range,rnames.range,price.range) {
  ws2018 <- gs_read(ss=ss,ws=ws,range=main.range,check.names=TRUE)
  
  # separate read for row names then
  rnames <- gs_read(ss=ss,ws=ws,range=rnames.range,col_names=FALSE)
  
  # read shipping cost
  shippedprice <- as.numeric(gs_read(ss=ss,ws=ws,range=price.range,col_names=FALSE))
  
  # convert to data frame. assigning row.names here doesn't work either
  bikes <- as.data.frame(ws2018,row.names=rnames$X1)
  # separate row.names call
  row.names(bikes) <- rnames$X1
  
  # fill in missing bike name. ie merged cell multiple models in one wheel size
  xset = grep('X[0-9]{1,2}$',names(bikes))
  for (i in xset) {
    colnames(bikes)[i] <- colnames(bikes)[i-1]
  }
  # remove .1 duplicate bike names across eg wheel size.
  xset = grep('.[1|2|3]$',names(bikes))
  for (i in xset) {
    colnames(bikes)[i] <- sub('.1','',colnames(bikes)[i]) 
  }
  # remove leading X for numeric model name
  xset = grep('^X[1-9]',names(bikes))
  for (i in xset) {
    colnames(bikes)[i] <- substring(colnames(bikes)[i],2) 
  }
  # convert the '.' back to ' '
  colnames(bikes) <- gsub('.',' ',colnames(bikes),fixed=TRUE)
  
  # combine build id for duplicate names
  builds <- which(!is.na(bikes['build',]))
  for (i in builds) {
    colnames(bikes)[i] <- sprintf('%s %s',colnames(bikes)[i],bikes['build',i])
  }
  
  # google sheet is transposed from normal data.frame context
  # melt,cast could be used, but simple transpose for now
  # somehow t(data.frame) stops being a data frame?
  bikes = as.data.frame(t(bikes))
  
  # remove the now unneeded build column
  bikes$build <- NULL
  
  # quickly remove non-syntactic characterw
  colnames(bikes) <- make.names(colnames(bikes))
  # convert factor to numeric. why doesn't it read directly as numeric?
  bikes$price <- as.numeric(as.character(bikes$price...CAN.))
  bikes$price...CAN. <- NULL
  # remove any predicted weights
  bikes$weight..lb.[grep('*',bikes$weight..lb.,fixed=TRUE)] <- NA
  bikes$weight <- as.numeric(as.character(bikes$weight..lb.))
  bikes$weight..lb. <- NULL
  # strange blank in one cell of the 27.5 wheel.size
  bikes$wheel.tire <- as.factor(gsub(' 27','27',bikes$wheel.tire))
  # strange extra 0 in one cell of the 27.5 wheel size
  bikes$wheel.tire <- as.factor(gsub('.50','.5',bikes$wheel.tire))
  bikes$bob.drop <- as.numeric(as.character(bikes$bob.drop))
  bikes$rear.centre <- as.numeric(as.character(bikes$rear.centre))
  bikes$chainstay <- as.numeric(as.character(bikes$chainstay))
  bikes$wheelbase <- as.numeric(as.character(bikes$wheelbase))
  bikes$front.centre <- bikes$wheelbase - bikes$chainstay
  bikes$SA <- as.numeric(as.character(bikes$SA)) * pi/180
  bikes$HA <- as.numeric(as.character(bikes$HA)) * pi/180
  bikes$HA..deg. <- NULL
  # expand compound value
  bikes$stem <- trunc(as.numeric(as.character(bikes$stem..mm.deg.)))
  bikes$stemA <- (as.numeric(as.character(bikes$stem..mm.deg.))-bikes$stem)*100 * pi/180
  bikes$stem..mm.deg. <- NULL
  bikes$reach <- as.numeric(as.character(bikes$reach))
  bikes$stack <- as.numeric(as.character(bikes$stack))
  bikes$head.tube <- as.numeric(as.character(bikes$head.tube))
  bikes$crank.length <- as.numeric(as.character(bikes$crank.length..mm.))/10
  bikes$crank.length..mm. <- NULL
  bikes$travel <- as.numeric(as.character(bikes$travel..mm.))
  bikes$travel..mm. <- NULL
  bikes$toptube <- as.numeric(as.character(bikes$toptube..cm.))
  bikes$toptube..cm. <- NULL
  
  # adjust by approximate wheel radius. need better value for tire profile.
  bikes$standover <- as.numeric(as.character(bikes$standover)) - 14*2.54
  # fill in blanks and update values
  # create a shipped price column, for all but the 27.5 which are frame-only specs
  bikes$shippedprice <- bikes$price
  # bikes$shippedprice[which(bikes$wheel.tire != '27.5')] <- shippedprice
  # for some reason getting a blank in element 52 of the bike list ' 27.5' in stead of '27.5'
  bikes$shippedprice[grep('27.5|29.0',bikes$wheel.tire,invert=TRUE)] <- shippedprice
  # fill in blanks from merged cells across multiple versions of one bike. don't copy missing weight though
  current_bike = row.names(bikes[1,])
  for (i in 2:nrow(bikes)) {
    if (row.names(bikes[i,])==current_bike) {
      na.set = which(is.na(bikes[i,names(bikes) != 'weight']) & !is.na(bikes[i-1,names(bikes) != 'weight'])) 
      if (length(na.set)) {
        bikes[i,na.set] <- bikes[i-1,na.set]
      }
    }  
    # if also more than 1 bike from a manufacturer
    else if (is.na(bikes[i,"manufacturer"])) {
      na.set = which(is.na(bikes[i,names(bikes) != "weight"]) & !is.na(bikes[i-1,names(bikes) != "weight"])) 
      if (length(na.set)) {
        bikes[i,na.set] <- bikes[i-1,na.set]
      }
    }
    current_bike = row.names(bikes[i,])
  }
  # for any missing offsets use suntour value
  bikes$fork.offset[is.na(bikes$fork.offset)] <- 4
  # convert drive to 1x format
  bikes$drive <- substr(bikes$drive,1,2)  
  # switch to tube to avoid reserved word frame. consolidate the alloys for factor model
  bikes$tube <- bikes$frame
  bikes$tube <- as.factor(gsub('[6|7][0-9]{3}','Al',bikes$tube))
  bikes$tube <- as.factor(gsub('^Al$','Al s.g.',bikes$tube))
  tube.levels <- c("4130 b.","Al s.g.","Al b.","Al d.b.","Al t.b.","Ti","carbon")
  bikes$tube <- ordered(bikes$tube,levels=tube.levels)
  bikes$frame <-NULL
  
  # order these by approximate rank of quality level. condense into level equivalents sram/shimano
  rear.der.levels <- c("tourney","altus/X3","acera/X4","alivio","X7","deore/NX","SLX","Zee/GX","XT")
  # couple hard-coded swaps for non-category fits
  bikes$rear.der[grep('microshift',bikes$rear.der)] <- 'tourney'
  bikes$rear.der[grep('xtr',bikes$rear.der)] <- 'SLX'
  # awkward way of doing this instead of loop. 
  remap <- sapply(bikes$rear.der,grepl,rear.der.levels)
  remap <- unlist(apply(remap,2,function(x) which(x>0 )),use.names = FALSE)
  bikes.rear.der <- as.vector(bikes$rear.der)
  bikes.rear.der[1:length(remap)] <- rear.der.levels[remap]
  bikes$rear.der <- as.factor(bikes.rear.der)
  bikes$rear.der <- ordered(bikes$rear.der,levels=rear.der.levels)
  bikes$brake.type <- ordered(bikes$brake.type,levels=c("v-brake","mech.","hydraulic"))
  bikes$shift.type <- ordered(bikes$shift.type,levels=c('thumb','twist','trigger'))
  bikes$drive <- ordered(bikes$drive,levels=c('1x','2x','3x'))
  
  # separate the plus
  plus.bikes <- grep('+',bikes$wheel.tire,fixed=TRUE)
  bikes$plus <- as.factor('reg')
  levels(bikes$plus) <- c('reg','plus')
  bikes$plus[plus.bikes] <- 'plus'
  # somehow gsub loses the factor
  bikes$wheel.tire <- as.factor(gsub('+','',as.factor(as.character(bikes$wheel.tire)),fixed=TRUE))
  return(bikes)
}

# create separate data frame for geo only
# should standover be included?
get.geo = function(bikes,geo.list) {
  g = bikes[,geo.list]
  g$wheel.tire <- as.numeric(as.character(g$wheel.tire))
  g$wheel.tire.factor <- as.factor(g$wheel.tire)
  g$HA <- g$HA * 180/pi
  g$SA <- g$SA * 180/pi
  g$stem <- g$stem / 10
  # remove duplicate models
  g <- g[!duplicated(g),]
  return(g)
}

# read in google sheet
ss <- gs_title("Kids 24\" bikes")
ws = "2018"
# range form give unique column name error for some but not all merged cells
# use of check.names=TRUE is partly the solution
# wasn't able to get 1st column interpreted as row names here
main.range = "d3:ci35"
rnames.range = "a4:a35"
price.range = "d41:bu41"
model.anchor1 = "CW5"
model.anchor2 = "DB5"

body_measure <- data.frame(leg=76,torso=50,arm=60)

if (!exists('bikes')) {
  # read.sheet(main.range,rnames.range,price.range)
  bikes <- read.sheet(ss,ws,main.range,rnames.range,price.range)
  bikes.geo <- get.geo(bikes,c('reach','stack','chainstay','standover','bob.drop','stem','toptube','front.centre','wheel.tire','SA','HA'))
}
set24.geo <- which(bikes.geo$wheel.tire.factor == '24')
set26.geo <- which(bikes.geo$wheel.tire.factor == '26')
set27.geo <- which(bikes.geo$wheel.tire.factor == '27.5' | bikes.geo$wheel.tire.factor == '29')

# list for bike coords
bc <- vector("list",nrow(bikes))
set24 <- which(substr(bikes$wheel.tire,1,2)==24 & bikes$spring != 'rigid')
set24r <- which(substr(bikes$wheel.tire,1,2)==24 & !is.na(bikes$weight))
set26 <- which(substr(bikes$wheel.tire,1,2)==26 & !is.na(bikes$weight))
set27 <- which(substr(bikes$wheel.tire,1,2)==27 | substr(bikes$wheel.tire,1,2)==29)
setnw <- which((substr(bikes$wheel.tire,1,2)==24 | substr(bikes$wheel.tire,1,2)==26) & is.na(bikes$weight))
attr(set26,"col")<-"red"
attr(set24,"col")<-"blue"
attr(set24r,"col")<-"blue"
for (bike in c(set24r,set26,set27,setnw)) {
  # print(c('calc_com',row.names(bikes[bike,])))
  bike_xy <- bike_coords((bikes[bike,]),body_measure)
  bike_xy$com <- calc_com(bike_xy,body_measure)
  bc[[row.names(bikes[bike,])]] = bike_xy
}

###############
# sample plots
###############

# preliminary visualizations
# plot.new()
tplot <- ggplot(data=bikes,aes(x=rear.der,y=price)) + geom_boxplot()
# listing the tplot object on the console command line, plots the plot. but, while a tplot
# statement in the script throws no error when the script is sourced, the plot is not plotted.
# requires an explicit plot statement
plot(tplot)

# model wheelbase with front.centre, rear.centre
lme.model <- lmList(data=bikes,front.centre~chainstay|wheel.tire)
# summary output of lme4 is dense. har-coded to pick out the coefficients without knowing how to get by name
lme.summary <- as.data.frame(summary(lme.model)[4])
cc <- data.frame(sl=lme.summary[,5],int=lme.summary[,1],wheel.tire=levels(bikes$wheel.tire))
rplot <- ggplot(data=bikes[c(set24r,set26,set27),],aes(x=chainstay,y=front.centre,col=wheel.tire)) + geom_point() +
  # don't get color aesthetic doing this directly, but don't get error either, but via data frame it works.
  # geom_abline(slope=lme.summary[,5],intercept=lme.summary[,1],aes(col=c(levels(bikes$wheel.tire)))) +
  geom_abline(data=cc, aes(slope=sl, intercept=int,col=wheel.tire))
plot(rplot)

# try kmeans clustering. 
m = as.matrix(cbind(bikes[c(set24r,set27),]$chainstay,bikes[c(set24r,set27),]$front.centre),ncol=2)
kls <- data.frame(k=3:4) %>% group_by(k) %>% do(kls=kmeans(m, .$k))
clusters <- kls %>%
  group_by(k) %>%
  do(tidy(.$kls[[1]]))
assignments <- kls %>%
  group_by(k) %>%
  do(augment(.$kls[[1]], m))
clusterings <- kls %>%
  group_by(k) %>%
  do(glance(.$kls[[1]]))
p1 <- ggplot(assignments, aes(X1, X2)) +
  geom_point(aes(color=.cluster)) +
  coord_equal(ratio=1) +
  facet_wrap(~ k)
# p1
# strange. the augment have columns X1,X2 while the tidy have x1,x2. the aes inherits, but just overwrite it. 
# y=, x= apparently optional.
p2 <- p1 +
  geom_point(data = clusters, aes(x1,x2),size = 10, shape = "x")
plot(p2)

# plot the sum of squares assessment of the cluster
p2<-ggplot(clusterings, aes(k, betweenss/totss*100)) + geom_line() +
  scale_y_continuous(limits=c(0,100),expand = c(0,0))
# plot(p2)
# the chainstay frontcentre data don't cluster according to wheelsize. they do cluster in a way that 
# suggests more variabiliitky in front.centre than rear but nothing obvious.

# try clustering in the higher dimension of the full geo
m <- as.matrix(bikes.geo[c(set24.geo,set26.geo,set27.geo),colnames(bikes.geo)!='wheel.tire.factor'])
kl <- kmeans(m,6)
# plot in reduced dimension with pca. clusplot is not ggplot so have no facet_wrap, and, 
# clusplot is not a grob, so can't use with gridExtra pacakge either. try to kludge with gridGraphics, 
# somehow it grabs an existing plot and makes a grob out of it?
pcolor = as.numeric(bikes.geo[c(set24.geo,set26.geo,set27.geo),'wheel.tire.factor'])
clusplot(m,kl$cluster,labels=2,
         col.p=pcolor)
kls2 <- data.frame(k=3:6) %>% group_by(k) %>% do(kls2=kmeans(m,.$k))
# the result of this pipe has the following syntax to access the cluster vec: kls$kls[[1]]$cluster 
# thought to pull out the cluster vectors and plot manually, but finally got the clusplot to work
# see below
kls.clustervec <- data.frame(k=3:4) %>% group_by(k) %>% do(kls=kmeans(m,.$k)$cluster)

# plot function returns grob to list for use with grid.arrange 
plot.single = function(m,clster,pcolor,tcolor,main,lbls,clus.color,xlab,ylab) {
  clusplot(m,clster, col.p=pcolor, main=main, sub=NULL, lines=0, labels=lbls, col.clus=clus.color,xlab=xlab,ylab=ylab,col.txt=tcolor)
  cp <- grab_grob()
  return(cp)
}
# closure version of the plot func for use with murrell variant
plot.single.closure = function(m,clster,pcolor,tcolor,main,lbls,clus.color,xlab,ylab) {
  function() {
    par(mar=c(7.2, 7.2, 1, 1), mex = .3, tcl = .15, mgp = c(3, .15, 0))
    clusplot(m,clster, col.p=pcolor, main=main, sub=NULL, lines=0, labels=lbls, col.clus=clus.color,
             xlab=xlab,ylab=ylab,col.txt=tcolor,cex.txt=0.75)
# this broke the functionality somehow
#    cp <- grid.grab()
#    return(cp)
  }
}
# use grid.grab to create the plot object
grab_grob <- function(){
  grid.echo()
  grid.grab()
}
p=list()
for (i in 1:2) {
  if (i==1) {
    main <- 'principal geometry components'
    lbls <- 3
    col.clus="white"
    col.pt = "white"
    col.text = pcolor
    xlab <- 'V1'
    ylab <- 'V2'
  } else {
#      main <- NULL
    lbls <- 0
    col.clus="black"
    col.pt = pcolor
    xlab <- NULL
    ylab <- NULL
  }
  p[[i]] <- plot.single(m,kls2$kls2[[i]]$cluster,col.pt,col.text,main,lbls,col.clus,xlab,ylab)
}

# grid.arrange with list of grob had unworkable overlap issues. or maybe par was needed
# grid.arrange(grobs=p)

# instead try the murrell variation to avoid grid.arrange and use plot function as an arg of grid.echo
# no idea if the viewports are needed or not, they are just copied
# still have a slight problem with spacing the main title of the chart, or top chart if more than one, 
# is off the top so not using main titles. or maybe different par needed?
grid.newpage()
pushViewport(viewport(layout = grid.layout(1, 1, 
                                           widths = unit(c(1), "null"),
                                           heights = unit(c(0.9), "null"))))
pushViewport(viewport(layout.pos.col = 1, layout.pos.row = 1))
lbls <- 3
col.clus="white"
col.pt = "white"
main<-NULL
pf2 <- plot.single.closure(m,kls2$kls2[[1]]$cluster,'white',pcolor,main,lbls,col.clus,'V1','V2')
# how to make a legend?
# legend('bottomright',legend=c(levels(bikes$wheel.tire))) # didn't work
grid.echo(pf2, newpage = FALSE)
# for side legend, use 2 column layout above
# upViewport()
# pushViewport(viewport(layout.pos.col=2,layout.pos.row=1))
# inset legend. very awkward. the 0,0 edge of the parent viewport is way off the bottom left corner of the graphics
# device. have to position by trial and error? also, can't quite seem to see any way to get color text in the grid.legend
# function, would have to make a legend manually to get the color text. 
pushViewport(viewport(x=.15,y=.2,width=0.15,height=0.15))
grid.legend(levels(bikes$wheel.tire),pch=16,vgap=0.5,gp=gpar(col=1:4),draw=TRUE)
upViewport()
# not sure how to print this to a file using grid.echo. could be that grid.echo|plot could be a function arg to 
# the plotprint function. but, for multiple viewports how would that work. save to file manually in this case

# more generally, look at the princomps
res.pca <- prcomp(m,TRUE)
fviz_eig(res.pca)
fviz_pca_ind(res.pca,col.ind='cos2')
fviz_pca_var(res.pca,col.var='contrib')

# plot pairs of geometries
bike.list = c('Vertex','YamaJama 24','Cujo')
bike.list2 = c('soul XS','YamaJama 26','Lookout')
bike.list3 = c('Edge 24','Edge 26')
bike.list = c('YamaJama 24','YamaJama 26')
bike.list = c('soul L','rockhopper L','Procal AL L','revolver L')
bike.list = c('405','Youngster 100','Superfly 26','MX Team')
bike.list = c('403','Scout 26','Timber XT','Bonaly 26')
plot.obj <- plot_bike(bike.list,bc,FALSE)
if (PLOT.PRINT) {
  plotprint(plot.obj,'geo.plot.26',512,512,TRUE)
}

grid.newpage()
pushViewport(viewport(layout = grid.layout(4, 1, 
                                           widths = unit(c(1), "null"),
                                           heights = unit(c(1, 1,1,1), "null"))))
for(i in 1:4) {
  pushViewport(viewport(layout.pos.col = 1, layout.pos.row = i))
  if (i==1) {
    main <- NULL
    xlab <- 'V1'
    ylab <- 'V2'
  } else {
    main <- NULL
    col.pt = pcolor
    xlab <- ''
    ylab <- ''
  }
  lbls=0
  col.pt = pcolor
  col.clus = "black"
  pf2 <- plot.single.closure(m,kls2$kls2[[i]]$cluster,col.pt,col.text,main,lbls,col.clus,xlab,ylab)
  grid.echo(pf2, newpage = FALSE)
  upViewport()
}

upViewport()


# use a scatter plot matrix as a quicklook at everything
pairs(bikes[,c('tube','weight','crank.length','HA','SA')])

# continue searching for trends in the wheel size that would match with long/low/slack
# bob.drop should be correlated with reach for long/low, reach/stem, reach/ha
lme.model <- lmList(data=bikes,bob.drop~front.centre|wheel.tire)
# summary output of lme4 is dense. har-coded to pick out the coefficients without knowing how to get by name
lme.summary <- as.data.frame(summary(lme.model)[4])
tplot <- ggplot(data=bikes[c(set26,set27,set24r),],aes(x=front.centre,y=-bob.drop,col=wheel.tire)) + geom_point() +
  geom_text(aes(label=rownames(bikes[c(set26,set27,set24r),])),position=position_jitter(width=.2,height=.3)) +
  labs(y='bob drop (cm)')
plot(tplot)

# reach/HA
lme.model <- lmList(data=bikes,HA~reach|wheel.tire)
# summary output of lme4 is dense. har-coded to pick out the coefficients without knowing how to get by name
lme.summary <- as.data.frame(summary(lme.model)[4])
tplot <- ggplot(data=bikes[c(set26,set24r,set27),],aes(x=reach,y=HA*180/pi,col=wheel.tire)) + geom_point() +
  labs(y="HA (deg)") +
  geom_text(aes(label=rownames(bikes[c(set26,set24r,set27),])),position=position_jitter(width=.2,height=.3))
plot(tplot)



# another attempt at clustering with cmdscale multi-di scaling
# weight/normalize for differing units?
# optionally normalize by standard dev
bikes.geo.std <- bikes.geo[,!colnames(bikes.geo) %in% c('wheel.tire','wheel.tire.factor')]
v<-sqrt(diag(var(bikes.geo.std)))
mn <- colMeans(bikes.geo.std)
bikes.geo.std <- bikes.geo.std - mn[col(bikes.geo.std)]
bikes.geo.std <- bikes.geo.std / v[col(bikes.geo.std)]
mds <- as.data.frame(cmdscale(dist(bikes.geo.std)))
vplot <- ggplot(data=mds,aes(x=V1,y=V2,col=bikes.geo$wheel.tire.factor)) + 
  geom_text(aes(label=rownames(bikes.geo)),position=position_jitter(height=.1))
vplot$labels$colour <- "wheel"
plot(vplot)
if (PLOT.PRINT) {
  plotprint(vplot,'geo.pca',512,512,TRUE)
}

# facetted multi-plot for pairs of geo. 
# merge did some sort of cross product, try cbind
# bikes.geo.melt <- merge(melt(bikes.geo,measure.vars=c('front.centre','HA','reach'),id.vars='wheel.tire.factor'),
#                        melt(bikes.geo,measure.vars=c('bob.drop','stem','stack'),id.vars=c('wheel.tire.factor')),by="wheel.tire.factor")
# still very clunky assembly, must manually remove the duplicated id.vars
bikes.geo.melt <- cbind(melt(bikes.geo,measure.vars=c('front.centre','HA','reach','toptube'),id.vars='wheel.tire.factor',variable.name='key1',value.name='val1'),
                        melt(bikes.geo,measure.vars=c('bob.drop','stem','stack','SA'),id.vars='wheel.tire.factor',variable.name='key2',value.name='val2'))
bikes.geo.melt <- bikes.geo.melt[,!duplicated(colnames(bikes.geo.melt))]
bikes.geo.melt$key <- paste(bikes.geo.melt$key1,bikes.geo.melt$key2,sep='-')
bikes.geo.melt[,c('key1','key2')] <- NULL
# try gather instead.similar to melt, but can't seem to allow for a aes color column like wheel.tire.factor
bikes.geo.melt2 <- cbind(gather(bikes.geo[,c('stem','reach','bob.drop','toptube')],key1,val1),gather(bikes.geo[,c('HA','stack','front.centre','SA')],key2,val2))
bikes.geo.melt2$key <- paste(bikes.geo.melt2$key1,bikes.geo.melt2$key2,sep='-')
bikes.geo.melt2[,c('key1','key2')] <- NULL
# make the plot
wplot <- ggplot(data=bikes.geo.melt,aes(x=val1,y=val2,col=wheel.tire.factor)) + geom_point() +
  facet_wrap(~key,nrow=1,scales="free")
wplot$labels$colour <- 'wheel'
plot(wplot)
if (PLOT.PRINT) {
  plotprint(wplot,'geo-pairs-facet',1536,384,TRUE)
}
##########################################
# models 
###########################################

# A. price with weight + categoricals

# A1. limited set with rear der

# adding tubeset factor reduced adjusted R2 from 9661 to 9612. it appears to have no relation to price
lm.model.A1 <- lm(data=bikes[c(set24r,set26),],formula=log10(shippedprice)~weight*spring*wheel.tire*rear.der,weights=(shippedprice))
cip <- data.frame(predict(lm.model.A1))
predicted.price.lm.model.A1 <- data.frame(yhat=10^(cip$predict.lm.model.A1.),y=bikes[c(set24r,set26),'shippedprice'],x=bikes[c(set24r,set26),'weight'])
predicted.price.lm.model.A1$bike <- row.names(bikes[c(set24r,set26),])
# due to limited bikes and wide range of derailleur, ignore all bikes for which the model fits exactly.
model.set <- which(abs(predicted.price.lm.model.A1$yhat-predicted.price.lm.model.A1$y) >0.0001)
# redo the fit with this subset. need this interim set to keep the correct indexing
bikes.2426 = bikes[c(set24r,set26),]
lm.model.A1 <- lm(data=bikes.2426[model.set,],formula=log10(shippedprice)~weight*spring*wheel.tire*rear.der,weights=(shippedprice))
cip <- data.frame(predict(lm.model.A1))
predicted.price.lm.model.A1 <- data.frame(yhat=10^(cip$predict.lm.model.A1.),y=bikes.2426[model.set,'shippedprice'],x=bikes.2426[model.set,'weight'])
predicted.price.lm.model.A1$bike <- row.names(bikes.2426[model.set,])
predicted.price.lm.model.A1$res <- predicted.price.lm.model.A1$yhat - predicted.price.lm.model.A1$y

# can't use the lmodel's own residuals column directly due to the log10, calculate rmse explicitly
rmse <- round(sqrt(sum((predicted.price.lm.model.A1[,'yhat'] - predicted.price.lm.model.A1[,'y'])^2) / length(model.set)))
# mean error comes in lower due to outliers getting squared up
me <- round((sum(abs(predicted.price.lm.model.A1[,'yhat'] - predicted.price.lm.model.A1[,'y'])) / length(model.set)))
rplot <- ggplot(data=predicted.price.lm.model.A1,aes(x=x,y=yhat,col="model",shape="model")) + geom_point() +
  geom_point(data=predicted.price.lm.model.A1,aes(x=x,y=y,shape="true",col="true")) +
  guides(colour=FALSE) +
  scale_shape_manual(values=c(0,3)) + 
  labs(y="price",x="weight",title="w/derailleur") +
  # can't figure out the aes for geom_text to work properly
  # geom_text(aes(x=30,y=3000,col="blue"),label=sprintf("rmse = $%d",rmse),inherit.aes=FALSE) +
  annotate("text",x=26,y=3000,label=sprintf("mean dev. = $%d",me)) +
  # expand c(0,0) is needed to trim the padding?
  scale_x_continuous(limits=c(18,31),expand = c(0,0)) + scale_y_continuous(limits=c(0,3500),expand = c(0,0))
plot(rplot)
if (PLOT.PRINT) {
  plotprint(rplot,'lm-price-wder',512,512,TRUE)
}

# plot the sorted residual. 
r2plot <- ggplot(data=predicted.price.lm.model.A1,aes(x=reorder(bike,predicted.price.lm.model.A1$res),y=res))  + 
  geom_col() + coord_flip() +
  labs(y='residual',x=NULL)
plot(r2plot)

# r eload to spreadhseet. note should preclear the full possible range in spreadsheet
# otherwise, changing models to affect different numbers of bikes in script could result in 
# uncleared data that isn't obvious in the spreadsheet
# short of deleting an entire sheet, googlesheets doesn't seem to be able to clear cells.
# gs_edit_cells can only upload a value. attempt to create a data.frame of NULL, which in 
# R is an empty object, literally uploads the string NULL to spreadsheet. still, the unused
# cells now populated with NA or NULL resolve the ambiguity and can at least manually be cleared.
GS_UPLOAD=FALSE
GS.NULL = matrix(nrow = nrow(bikes), ncol =4)
if (GS_UPLOAD==TRUE) {
  gs_edit_cells(ss=ss,ws=ws,input=GS.NULL,anchor="CV5",col_names=FALSE)
  gs_edit_cells(ss=ss,ws=ws,input=predicted.price.lm.model.A1[order(-predicted.price.lm.model.A1$res),c('bike','y','yhat','res')],anchor="CV5",col_names=FALSE)
}
###################################
# A2. all bikes, with-out rear der
###################################
rm(rplot)
lm.model.A2 <- lm(data=bikes[c(set24r,set26),],formula=log10(shippedprice)~weight*spring*wheel.tire,weights=(shippedprice))
# predict without newdata should give the values at the fitted data points without having
# to reconstruct a dummy data.frame. but doesn't work?
cip <- data.frame(predict(lm.model.A2))
predicted.price.lm.model.A2 <- data.frame(yhat=10^(cip$predict.lm.model.A2.),y=bikes[c(set24r,set26),'shippedprice'],x=bikes[c(set24r,set26),'weight'])
model.set <- c(set24r,set26)
predicted.price.lm.model.A2$res <- predicted.price.lm.model.A2$yhat - predicted.price.lm.model.A2$y
predicted.price.lm.model.A2$bike <- row.names(bikes[c(set24r,set26),])
RSS <- sqrt(c(crossprod(10^lm.model.A2$residuals))/length(lm.model.A2$residuals))
rmse <- round(sqrt(sum((predicted.price.lm.model.A2[model.set,'yhat'] - predicted.price.lm.model.A2[model.set,'y'])^2) / length(model.set)))
me <- round((sum(abs(predicted.price.lm.model.A2[,'yhat'] - predicted.price.lm.model.A2[,'y'])) / length(model.set)))
rplot <- ggplot(data=predicted.price.lm.model.A2[model.set,],aes(x=x,y=yhat,shape="model",col="model")) + geom_point() +
  geom_point(data=predicted.price.lm.model.A2[model.set,],aes(x=x,y=y,col="true",shape="true")) +
  scale_shape_manual(values=c(0,3)) +
  guides(colour=FALSE) +
  labs(y="price",x="weight",title="w/out deraileur") +
  # can't figure out the aes for geom_text to work properly
  # geom_text(aes(x=30,y=3000,col="blue"),label=sprintf("rmse = $%d",rmse),inherit.aes=FALSE) +
  annotate("text",x=26,y=3000,label=sprintf("mean dev. = $%d",me)) +
  # expand c(0,0) is needed to trim the padding?
  scale_x_continuous(limits=c(18,31),expand = c(0,0)) + scale_y_continuous(limits=c(0,3500),expand = c(0,0))
plot(rplot)
if (PLOT.PRINT) {
  plotprint(rplot,'lm-price',512,512,TRUE)
}
if (GS_UPLOAD==TRUE) {
  gs_edit_cells(ss=ss,ws=ws,input=GS.NULL,anchor="DA5",col_names=FALSE)
  gs_edit_cells(ss=ss,ws=ws,input=predicted.price.lm.model.A2[order(-predicted.price.lm.model.A2$res),c('bike','y','yhat','res')],anchor="DA5",col_names=FALSE)
}

# plot the sorted residual. 
r2plot <- ggplot(data=predicted.price.lm.model.A2,aes(x=reorder(bike,predicted.price.lm.model.A2$res),y=res))  + 
  geom_col() + coord_flip() +
  labs(y='residual',x=NULL)
plot(r2plot)

##########################################
# B. weight with price and categoricals
##########################################
# b.1 

lm.model.B1 <- lm(data=bikes[c(set24r,set26),],formula=weight ~ log10(shippedprice)+tube+rear.der+brake.type+wheel.tire+
                    spring+shift.type+plus)
cip <- data.frame(predict(lm.model.B1,interval="confidence"))
cip$weight <- bikes[c(set24r,set26),'weight']
cip$res <- cip$fit - cip$weight
cip$price <- bikes[c(set24r,set26),'shippedprice']
me <- sum(abs(cip$res))/nrow(cip)
# attempt new predict
set.noweight <- which(is.na(bikes$weight) & bikes$wheel.tire != "27.5" & bikes$wheel.tire != "29.0")
df.noweight <- data.frame(bikes[set.noweight,c('wheel.tire','shippedprice','tube','rear.der','brake.type','spring','drive','shift.type','plus')])
# temp remap factor Ti since no predictive data.
df.noweight$tube <- gsub("Ti","carbon",df.noweight$tube)
df.noweight$tube <- gsub("4130 b.","Al s.g.",df.noweight$tube)
cip.noweight <- data.frame(predict(lm.model.B1,newdata = df.noweight,interval="confidence"))
cip.noweight$price <- bikes[set.noweight,'shippedprice']
print(cip.noweight)

yplot <- ggplot(data=cip,aes(x=price,y=fit,ymin=lwr,ymax=upr),col="model") + geom_errorbar(width=10) +
  geom_point(aes(x=price,y=weight,col="true")) + 
  labs(y="weight (lb)") +
  annotate("text",y=29,x=2000,label=sprintf('mean dev.=%1.2f',me))
yplot <- yplot + geom_errorbar(data=cip.noweight,aes(x=price,ymin=lwr,ymax=upr),col="blue",width=10) +
  geom_text(data=cip.noweight,aes(x=price,y=fit,label=rownames(cip.noweight)),size=3,col="blue")
plot(yplot)
if (PLOT.PRINT) {
  plotprint(yplot,'weight-price',1024,256,TRUE)
}

# upload predicted weight to  spreadsheet
GS_UPLOAD_PRED=FALSE
if (GS_UPLOAD_PRED) {
  for (i in seq(1,length(set.noweight))) {
    ii = set.noweight[i]+3
    if (ii<=26) { 
      col.letter = toupper(letters[ii])
    } else if (ii <= 52) { 
      col.letter = sprintf('A%s',toupper(letters[ii-26])) 
    } else { 
      col.letter = sprintf('B%s',toupper(letters[ii-52])) 
    }
    print(c(cip.noweight$predict.lm.model.B1..newdata...df.noweight.[i],sprintf('%s7',col.letter)))
    gs_edit_cells(ss=ss,ws=ws,input=sprintf('*%2.1f*',cip.noweight$predict.lm.model.B1..newdata...df.noweight.[i]),anchor=sprintf("%s7",col.letter))
    
  }
  
}

# linear modelling with weight only
# sometimes plot.new() is required by error message, but when run-sourcing the script, it creates
# a blank plot.
# plot.new()
# plot the raw data
splot <- ggplot(data=bikes[c(set26,set24r),],aes(x=weight,y=shippedprice,col=wheel.tire)) + 
  geom_point(aes(shape=wheel.tire,size=spring)) +
  labs(y="price") + 
  scale_color_manual(values=c('24'='blue','24+'='blue','26'='red')) +
  scale_shape_manual(values=c('24'=16,'24+'=17,'26'=16)) +
  scale_size_manual(values=c('air'=2,'coil'=1,'rigid'=3)) +
  # expand c(0,0) is needed to trim the padding?
  scale_x_continuous(limits=c(19,31),expand = c(0,0)) + 
  scale_y_continuous(limits=c(0,3500),expand = c(0,0))


# confidence interval for nls is highly skewed. use log-linear model instead, weights are needed
# to compensate for the log scaling in the least-squares minimization
model.type <- "LOGLIN"
if (model.type=="LOGLIN") {
  for (set in list(set26,set24r)) {

    if (isTRUE(all.equal(set,set24r))) {
        # weight and spring mmodel only:
        lm.model <- lm(data=bikes[set,],formula=log10(shippedprice)~weight*spring,weights=(shippedprice))
        for (spr in c("air","coil","rigid")) {
          fit.wts <- data.frame(weight=seq(20,30,length.out=100), spring=spr)
          cip <- data.frame(predict(lm.model,newdata=fit.wts,interval="confidence"),x=fit.wts$weight)
          price.lm.model <- data.frame(y=10^(cip$fit),lwr=10^(cip$lwr),upr=10^(cip$upr),x=fit.wts$weight)
          # construct the confidence interval manually, since geom_smooth can't work around the log
          splot <- splot + geom_line(data=price.lm.model,aes(x=x,y=y),col=attr(set,"col")) +  
            geom_ribbon(data=price.lm.model,aes(x=x,ymin=lwr,ymax=upr),alpha=0.1,inherit.aes=FALSE)
        }
      } else {
    # add regressor?
        lm.model <- lm(data=bikes[set,],formula=log10(shippedprice)~weight,weights=(shippedprice))
        fit.wts <- data.frame(weight=seq(20,30,length.out=100))
        cip <- data.frame(predict(lm.model,newdata=fit.wts,interval="confidence"),x=fit.wts$weight)    
        price.lm.model <- data.frame(y=10^(cip$fit),lwr=10^(cip$lwr),upr=10^(cip$upr),x=fit.wts$weight)
        # note: creating a range for prediction from the model, the column name of the data frame must be the same as 
        # the variable used in the model: wts
        # construct the confidence interval manually, since geom_smooth can't work around the log
        splot <- splot + geom_line(data=price.lm.model,aes(x=x,y=y),col=attr(set,"col")) +  
          geom_ribbon(data=price.lm.model,aes(x=x,ymin=lwr,ymax=upr),alpha=0.1,inherit.aes=FALSE)
      }
    
  } 
} else {
  # non-linear model
  # again note use of variable name wts in the model, to match the column name wts, in the prediction range
  exp.model <- nls(y ~ a*exp(b*wts), data=data.frame(y=price,wts=wts), start=list(a=10000,b=-.001))
  price.exp.model <- data.frame(y=predict(exp.model,newdata=fit.wts),x=fit.wts$wts)

  # can't use geom_smooth directly with nls, no confidence estimates. have to create it all manually  
  # splot <- splot + geom_line(data=price.exp.model,aes(x=x,y=y))
  splot <- splot + geom_line(data=price.exp.model,aes(x=x,y=y),col=attr(set,"col"))
  # splot <- splot + geom_smooth(method="glm",family=gaussian(link="identity"))
  
  # fit parameter uncertainties can be obtained from confint2
  ci <- confint2(exp.model)
  # confidence intervals for non-linear are estimated by monte carlo bootstrap
  # currently, for bikes dataset, this estimate is returning a very skewed interval relative
  # to the fit value. 
  # predictNLS quantile function returns column names that are ##%, both of whidh
  # are incompatible with geom_line. 
  # cip <- as.data.frame(predictNLS(exp.model,newdata=fit.wts,level=0.9))
  # rather than as.data.frame, data.frame corrects for the illegal column 
  # name by creating a compatible name from the ##%,replacing with X##. 
  cip <- data.frame(predictNLS(exp.model,newdata=fit.wts,level=0.9,nsim=10000),x=fit.wts$wts)
  # remove the . from the created compatible name. no \ in R regex, use fixed=TRUE
  # to interpret . as a literal.
  colnames(cip) <- sub('.','',colnames(cip),fixed=TRUE)
  cip$x <- fit.wts$wts
  # aes is inherited. because the previous data have a wheel.tire color, and there is no wheel.tire in this
  # data frame, creates an object not found error unless inherit.aes FALSE
  splot <- splot + geom_line(data=cip,aes(x=x,y=X95),inherit.aes=FALSE) + geom_line(data=cip,aes(x=x,y=X5),inherit.aes=FALSE)
  
}
plot(splot)

