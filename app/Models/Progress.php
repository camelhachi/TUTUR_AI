<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Progress extends Model
{
    use HasFactory;

    protected $primaryKey = 'progress_id';
    protected $table = 'progress';
    protected $fillable = [
        'progress_id',
        'lesson_id',
        'user_id',
        'is_completed', 
    ];

    protected $casts = [
        'is_completed' => 'boolean', 
    ];

    public function lesson()
    {
        return $this->belongsTo(Lesson::class, 'lesson_id', 'lesson_id');
    }

    /**
     * Get the user associated with the progress.
     */
    public function user()
    {
        return $this->belongsTo(User::class, 'user_id', 'user_id');
    }
    public $timestamps = false;
}
