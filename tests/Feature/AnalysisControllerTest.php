<?php

namespace Tests\Feature;

use App\Services\PredictionRunner;
use Illuminate\Http\UploadedFile;
use Illuminate\Support\Facades\Storage;
use Tests\TestCase;

class AnalysisControllerTest extends TestCase
{
    public function test_each_file_receives_enough_execution_time_and_worker_errors_return_as_json(): void
    {
        Storage::fake('local');
        config(['ml.timeout' => 180, 'ml.keep_uploads' => false]);
        $originalLimit = (int) ini_get('max_execution_time');
        $observedLimits = [];
        $runner = $this->mock(PredictionRunner::class);
        $runner->shouldReceive('candidates')->andReturn([['python'], ['python3']]);
        $runner->shouldReceive('run')->twice()->andReturnUsing(
            function (string $path, string $name) use (&$observedLimits): array {
                $observedLimits[] = (int) ini_get('max_execution_time');
                set_time_limit(120);

                return ['ok' => false, 'file' => $name, 'error' => 'Timed out after 180s.'];
            }
        );

        try {
            set_time_limit(120);

            $response = $this->postJson('/analyse', ['files' => [
                UploadedFile::fake()->createWithContent('first.csv', "rate,label\n1,0\n"),
                UploadedFile::fake()->createWithContent('second.csv', "rate,label\n2,1\n"),
            ]]);

            $response->assertOk()->assertJsonPath('results.0.error', 'Timed out after 180s.')
                ->assertJsonPath('results.1.error', 'Timed out after 180s.');
            $this->assertSame([270, 270], $observedLimits);
            $this->assertSame([], Storage::disk('local')->allFiles());
        } finally {
            set_time_limit($originalLimit);
        }
    }
}
