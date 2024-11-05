<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\User;
use Illuminate\Support\Facades\Session;

class userController extends Controller
{
    public function createUser(Request $request)
    {

        // Retrieve the user input
        $name = $request->input('name');
        $username = $request->input('username');
        $email = $request->input('email');
        $password = $request->input('password');

        // create user in db
        $user = new User();
        $user->full_name = $name;
        $user->username = $username;
        $user->email = $email;
        $user->password = $password;
        $user->save();

        // You can return the message to the view or handle it as needed
        return view('login');
    }

    public function loginUser(Request $request)
    {
        // Retrieve the user input
        $username = $request->input('username');
        $password = $request->input('password');

        // Find the user in the database by username
        $user = User::where('username', $username)->first();

        // Check if the user exists and the password matches
        if ($user && $user->password == $password) {

            session(['user_id' => $user->user_id]);

            // If the login is successful, return the dashboard view
            return redirect()->route('dashboard.page');
        } else {
            // If the login fails, you can return to the login page with an error message
            return redirect()->back()->withErrors(['Invalid username or password']);
        }
    }

    public function logoutUser(Request $request)
    {
        // Clear the session
        Session::forget('user_id');

        // Optionally, you can also flush all session data
        Session::flush();

        // Redirect to the login page
        return redirect()->route('login.page');
    }


    public function showProfile(Request $request)
    {
        $userID = $request->session()->get('user_id');
        $user = User::where('user_id', $userID)->first();

        return view('navbar', ['user' => $user]);
    }

    // public function showDashboard(Request $request)
    // {
    //     // Retrieve the user ID from the session
    //     $userID = Session::get('user_id');

    //     // Retrieve the user data from the database
    //     $user = User::find($userID);

    //     // Pass the user data to the dashboard view
    //     return view('dashboard', ['user' => $user]);
    // }


    public function showProfileDetails(Request $request)
    {
        $userID = Session::get('user_id');

        $user = User::find($userID);

        return view('profile', ['user' => $user]);
    }

    public function updateProfile(Request $request)
    {
        // Retrieve the user ID from the session
        $userID = Session::get('user_id');

        // Find the user in the database
        $user = User::find($userID);

        // Update the user data
        $user->full_name = $request->input('name');
        $user->email = $request->input('email');
        $user->save();

        // Redirect to the profile page with a success message
        return redirect()->route('dashboard.page')->with('success', 'Profile updated successfully');
    }
}
