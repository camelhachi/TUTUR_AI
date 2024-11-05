<?php

namespace App\Http\Controllers;

use App\Models\Task;
use Illuminate\Http\Request;
use App\Models\Progress;

class taskController extends Controller
{

    public function showAssesment(Request $request, $lesson_id, $progress_id)
    {

        $tasks = Task::where('lesson_id', $lesson_id)->get();
        

        return view('assesment', compact('lesson_id', 'progress_id','tasks'));

    }

    public function completeProgress(Request $request, $lesson_id, $progress_id)
    {
        // Retrieve the user ID from the session
        $userID = $request->session()->get('user_id');

        // Find the progress entry for the given progress ID, lesson ID, and user ID
        $progress = Progress::where('progress_id', $progress_id)
            ->where('lesson_id', $lesson_id)
            ->where('user_id', $userID)
            ->first();

        // Check if the progress entry exists
        if ($progress) {
            // Update the status to true
            $progress->is_completed = true;

            // Save the changes
            $progress->save();

            // Optionally, you can return a success message or redirect the user
            return redirect()->route('dashboard.page')->with('status', 'Progress updated successfully!');
        } else {
            // If the progress entry doesn't exist, return an error message
            return redirect()->back()->with('error', 'Progress entry not found.');
        }
    }
}
