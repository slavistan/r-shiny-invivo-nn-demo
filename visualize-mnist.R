library(keras)
library(shiny)
library(tidyverse)
 
# Load MNIST dataset Normalize and flatten 28x28 images using row-major layout (which is used by tensorflow).
# Also normalize the grayscale values into [0; 1] and one-hot-encode the labels.
mnist   <- dataset_mnist()
x.train <- array_reshape(mnist$train$x, c(nrow(mnist$train$x), 784)) / 255
x.test  <- array_reshape(mnist$test$x, c(nrow(mnist$test$x), 784)) / 255
y.train <- to_categorical(mnist$train$y, 10)
y.test  <- to_categorical(mnist$test$y, 10)

# Returns ggplot-heatmap from a row-major 1D vector. Kept generic enough to plot arbitrary 2D heatmaps, since ggplot
# does not have any heatmap geoms and 'image()' rotates the image for no healthy reason.
plot_heatmap <- function(data, nrow = 1, ncol = length(data)) {

  X = rep(1:ncol, times = nrow) # X = 123123123
  Y = rep(1:nrow, each  = ncol) # Y = 111222333

  heat.map <- ggplot()                                    +
    geom_raster(aes(x = X, y = Y, fill = data))           +
    coord_cartesian(xlim = c(1, nrow), ylim = c(1, nrow)) +
    theme_void()                                          +
    theme(legend.position="none")

  return(heat.map)
}

# Define user interface
ui <- fluidPage(
  fluidRow(
    column(4, offset=2, align="center",
      h2("Test Set"),
      sliderInput('test.index', label=NULL, min=1, max=10000, value=1000),
      strong(textOutput(outputId = 'label.test')),
      plotOutput(outputId = 'image.test')
    ),
    column(4, align="center",
      h2("Training Set"),
      sliderInput('train.index', label=NULL, min=1, max=60000, value=8400),
      strong(textOutput(outputId = 'label.train')),
      plotOutput(outputId = 'image.train')
    )
  )
)

# Define reactive outputs
server <- function(input, output) {

  # Pixmap of test image
  output$image.test <- renderPlot({

    img.out <- plot_heatmap(x.test[input$test.index,])

    # information can be passed explicitly via a list
    list(src = img.out)
  }, width = 200, height = 200 )
  
  # Pixmap of training image
  output$image.train <- renderPlot({

    plot_heatmap(x.train[input$train.index,])
  }, width = 200, height = 200)

  # Label of test image
  output$label.test <- renderText({

    return(paste0("Label: ", mnist$test$y[[input$test.index]]))
  })

  # Label of training image
  output$label.train <- renderText({

    return(paste0("Label: ", mnist$train$y[[input$train.index]]))
  })
}

# Run the app
app <- shinyApp(ui = ui, server = server)
