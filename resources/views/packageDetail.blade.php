<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="{{ asset('css/packageDetail.css') }}" />
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" />
    <title>packageDetail</title>
</head>

<body>
    <!-- Navbar -->
    <div class="navbar">
        <div class="nav-wrapper">
            <img src="pic.png" class="logo">
            <div class="nav2">
                <nav>
                    <ul>
                        <li><a href="{{ route('index.page') }}#home">Home</a></li>
                        <li><a href="{{ route('index.page') }}#about">About Us</a></li>
                        <li><a href="{{ route('index.page') }}#plan">Plans</a></li>
                    </ul>
                </nav>
                <button class="signup">Sign Up <span class="material-symbols-outlined">
                        trending_flat
                    </span></button>
            </div>
        </div>
    </div>
    <!-- Content -->
    <div class="container">
        <div class="wrapper">
            <div class="back"><span class="material-symbols-outlined">
                    keyboard_backspace
                </span><a href="{{ route('index.page') }}#plan">Back</a></div>

            <div class="detail">
                <div class="col1">
                    <img src="elements/Group 39494.png" alt="">
                </div>
                <div class="col2">
                    <div class="titlepkg">
                        <h1>Fluent Fingers</h1>
                        <h3>IDR 60K /Mo</h3>
                    </div>
                    <div class="content">
                        <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam consectetur, nunc pretium
                            sollicitudin blandit, elit leo efficitur velit, sit amet ultrices eros nisi a sem. Etiam a
                            felis
                            ornare, eleifend magna ut, feugiat orci. Pellentesque id suscipit turpis. Pellentesque
                            commodo
                            molestie fermentum. Etiam eget dapibus dui. Phasellus cursus lectus vel sem laoreet, nec
                            aliquam
                            dui facilisis. Nullam hendrerit dapibus mollis. Mauris vulputate turpis quis nibh bibendum,
                            sed
                            tristique ex egestas.</p>
                        <ul>
                            <li>Lorem ipsum dolor sit amet.</li>
                            <li>Sed a eros elementum, suscipit risus vitae.</li>
                            <li>Suspendisse tincidunt.</li>
                            <li>Praesent id mi eu sem egestas porta.</li>
                            <li>Sed fermentum metus eget.</li>
                            <li>Praesent consectetur magna finibus tincidunt commodo.</li>
                        </ul>
                    </div>
                    <button class="signup"> Purchase Now <span class="material-symbols-outlined">
                            trending_flat
                        </span></button>
                </div>
            </div>
        </div>

    </div>

</body>

</html>