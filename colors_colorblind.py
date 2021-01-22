import seaborn as sns


# #colors: Grey, Blue / dark blue, Light orange, Red, Yellow (me)
# #Renaud: grey, blue, light orange, pink, etc (Renaud)
# list_5_colors = ['Grey', 'Blue / Dark Blue', 'Light orange', 'Red', 'Yellow'] #useful for the alphas ([0.6,0.7,0.8,0.9,1])

# colors_alphas = {
#     0.6: 'Grey', 
#     0.7: 'Blue / Dark Blue', 
#     0.8: 'Light orange', 
#     0.9: 'Red', 
#     1: 'Yellow'
# }


current_palette = sns.color_palette('colorblind')
# colors_alphas = {
#     (0.6,0.6): current_palette[0], 
#     (0.7,0.7): current_palette[1], 
#     (0.8,0.8): current_palette[2], 
#     (0.9,0.9): current_palette[3], 
#     (1,1): current_palette[4]
# }
colors_alphas = {
    (0.6,0.6): current_palette[1], 
    (0.7,0.7): current_palette[0], 
    (0.8,0.8): current_palette[3], 
    (0.9,0.9): current_palette[8], 
    (1,1): current_palette[2]
}
