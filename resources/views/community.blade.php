<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="css/community.css" />
    <link rel="stylesheet"
        href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" />
    <title>Community</title>
</head>

<body>
    <!-- Navbar -->
    @include('navbar');
    <!-- Content -->
    <div class="content">
        <div class="container">
            <div class="col-1">
                <h2>Welcome to Tutur</h2>
                <h1>Community!</h1><br>
                <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque in feugiat mauris. Vivamus at ornare
                    mi. </p>

                <img src="elements/Orange Brown Simple Lined American Sign Language Portrait Educational Poster (2) 1.png"
                    alt="">
            </div>
            <div class="col-2">
                <div class="content1">
                    <div class="title-1">
                        <h1>Your community</h1>
                    </div>
                    <div class="card">
                        <div class="card-content">
                            <img src="elements/Pic1.jpeg" alt="">
                            <div class="card-text">
                                <h3>Binus University ASL Club </h3>
                                <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque in feugiat mauris.
                                    Vivamus at ornare mi. </p>
                                <a href="{{ route('communityDetail.page') }}"><span class="material-symbols-outlined">
                                    trending_flat
                                </span></a>

                            </div>
                        </div>
                    </div>
                </div>
                <div class="content2">
                    <div class="title-2">
                        <h1>Explore communities!</h1>
                    </div>
                    <div class="cards">
                        <div class="card">
                            <div class="card-content">
                                <img src="elements/Pic2.jpeg" alt="">
                                <div class="card-text">
                                    <h3>MediSign</h3>
                                    <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque in feugiat
                                        mauris.
                                        Vivamus at ornare mi. </p>
                                    <div class="view"><a href="">View</a><span class="material-symbols-outlined">
                                            trending_flat
                                        </span></div>
                                </div>
                            </div>
                        </div>
                        <div class="card">
                            <div class="card-content">
                                <img src="elements/Pic3.jpeg" alt="">
                                <div class="card-text">
                                    <h3>Magic Wands, Magic Hands</h3>
                                    <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque in feugiat
                                        mauris.
                                        Vivamus at ornare mi. </p>
                                    <div class="view"><a href="">View</a><span class="material-symbols-outlined">
                                            trending_flat
                                        </span></div>

                                </div>
                            </div>
                        </div>

                    </div>

                </div>
            </div>
        </div>
    </div>
    </div>
</body>